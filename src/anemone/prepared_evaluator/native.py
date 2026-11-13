"""Owned checkpoint reconstruction and exact native records, without a search runtime."""

# Native internal bindings intentionally retain runtime annotations.
# ruff: noqa: TC001
import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from random import Random
from typing import Any, cast

from anemone.checkpoints._protocols import IncrementalStateCheckpointCodec
from anemone.checkpoints.build_atoms import _serialize_branch_collection
from anemone.checkpoints.build_context import (
    _CheckpointBuildContext,
    _CheckpointBuildMetrics,
)
from anemone.checkpoints.build_evaluation_payloads import (
    _build_exploration_index_payload,
    _build_node_evaluation_payload,
)
from anemone.checkpoints.evaluation_payloads import (
    _restore_evaluation,
    _restore_exploration_index,
)
from anemone.checkpoints.io import (
    checkpoint_payload_to_jsonable,
)
from anemone.checkpoints.metadata import _validate_payload
from anemone.checkpoints.node_restore import (
    _build_tree,
    _create_nodes,
    _link_nodes,
    _restore_node_runtime_state,
    _restore_parent_order,
)
from anemone.checkpoints.payloads import (
    DeltaCheckpointStatePayload,
    SearchRuntimeCheckpointPayload,
    TreeCheckpointPayload,
)
from anemone.checkpoints.restore_atoms import _deserialize_branch
from anemone.checkpoints.rng_state import restore_random_state
from anemone.checkpoints.sharded_decode import (
    _algorithm_node_payload_from_mapping,
    _selector_payload_from_mapping,
    _tree_expansions_payload_from_mapping,
)
from anemone.checkpoints.state_handles import CheckpointBackedStateHandle
from anemone.checkpoints.state_handles_restore import (
    _create_state_handles,
    _create_state_resolver,
)
from anemone.checkpoints.tree_expansions_payloads import (
    _build_tree_expansions_payload,
    _restore_tree_expansions,
)
from anemone.checkpoints.value_serialization import serialize_checkpoint_atom
from anemone.indices.node_indices.factory import create_exploration_index_data
from anemone.indices.node_indices.index_types import IndexComputationType
from anemone.node_evaluation.evaluator_refresh import validate_refresh_node
from anemone.node_evaluation.full_refresh import iter_nodes
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.node_factory.algorithm_node_factory import AlgorithmNodeFactory
from anemone.node_factory.base import TreeNodeFactory
from anemone.node_selector.linoo import Linoo
from anemone.node_selector.opening_instructions import OpeningInstructor
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.objectives import SingleAgentMaxObjective
from anemone.tree_manager.tree_expander import TreeExpansions
from anemone.trees.tree import Tree

from .contracts import PreparationLimits, SupportedSearchRecipe, require
from .structure import DetachedStructure


def encode(value: object) -> bytes:
    """Encode auxiliary identity records, preserving native sequence/map ordering."""
    return json.dumps(
        checkpoint_payload_to_jsonable(value), separators=(",", ":"), allow_nan=False
    ).encode()


def decode(raw: bytes) -> SearchRuntimeCheckpointPayload:
    """Use existing native typed record decoders; no new checkpoint format."""
    data = json.loads(raw)
    return SearchRuntimeCheckpointPayload(
        format_version=data["format_version"],
        evaluator_version=data["evaluator_version"],
        rng_state=data["rng_state"],
        rollout_rng_state=data.get("rollout_rng_state"),
        tree=TreeCheckpointPayload(
            root_node_id=data["tree"]["root_node_id"],
            nodes=[
                _algorithm_node_payload_from_mapping(n) for n in data["tree"]["nodes"]
            ],
        ),
        selector_state=None
        if data["selector_state"] is None
        else _selector_payload_from_mapping(data["selector_state"]),
        latest_tree_expansions=None
        if data["latest_tree_expansions"] is None
        else _tree_expansions_payload_from_mapping(data["latest_tree_expansions"]),
    )


def validate_source(
    payload: SearchRuntimeCheckpointPayload,
    recipe: SupportedSearchRecipe,
    limits: PreparationLimits,
) -> None:
    """Admit complete layered DAG/codec ancestry and exact non-null derived coverage."""
    limits.check()
    _validate_payload(payload)
    require(
        recipe.depth_selection_policy
        in ("inverse_depth", "opened_count_depth_index", "alternating_by_step"),
        "unsupported Linoo policy",
    )
    require(recipe.branch_limit > 0, "invalid branch limit")
    require(
        payload.selector_state is not None,
        "uninitialized/absent selector state is unsupported in v1",
    )
    require(payload.rng_state is not None, "missing search RNG")
    require(
        (payload.rollout_rng_state is not None)
        == (recipe.rng_layout == "independent_rollout"),
        "RNG layout/coverage mismatch",
    )
    require(0 < len(payload.tree.nodes) <= limits.max_nodes, "node bound exceeded")
    records = {n.node_id: n for n in payload.tree.nodes}
    edges = sum(len(n.linked_children) for n in records.values())
    require(edges <= limits.max_branches, "branch bound exceeded")
    incoming: dict[int, list[int]] = {key: [] for key in records}
    for n in records.values():
        limits.check()
        require(
            n.evaluation is not None
            and n.exploration_index is not None
            and n.exploration_index.kind == "MinMaxPathValue",
            "missing evaluation/global-min-change coverage",
        )
        seen = set()
        for link in n.linked_children:
            branch = _deserialize_branch(link.branch_key)
            require(branch not in seen, "duplicate outgoing branch")
            seen.add(branch)
            require(
                link.child_node_id in records
                and records[link.child_node_id].depth == n.depth + 1,
                "graph is not a layered DAG",
            )
            incoming[link.child_node_id].append(n.node_id)
        state = n.state_payload
        if isinstance(state, DeltaCheckpointStatePayload):
            require(
                state.state_parent_node_id in records
                and records[state.state_parent_node_id].depth < n.depth,
                "invalid codec dependency ancestry",
            )
    for n in records.values():
        require(
            (not incoming[n.node_id]) == (n.node_id == payload.tree.root_node_id),
            "disconnected/root ancestry",
        )
        if n.parent_node_id is not None:
            require(
                n.parent_node_id in incoming[n.node_id],
                "representative graph parent mismatch",
            )
        if n.parent_node_order is not None:
            require(
                len(n.parent_node_order) == len(set(n.parent_node_order))
                and set(n.parent_node_order) == set(incoming[n.node_id]),
                "incomplete incoming parent order",
            )


class _NoOpening:
    """No opening or rollout capability is available to a detached selector."""

    def instructions_to_open_all_branches(self, *args: object, **kwargs: object) -> Any:
        msg = "prepared evaluation cannot perform Growth"
        raise TypeError(msg)


class _BudgetCodec:
    """Own one fresh inline codec and account every native state decode."""

    def __init__(
        self, codec: IncrementalStateCheckpointCodec[Any], limits: PreparationLimits
    ) -> None:
        self.codec = codec
        self.limits = limits
        self.count = 0

    def _tick(self) -> None:
        self.limits.check()
        self.count += 1
        require(
            self.count <= self.limits.max_state_decodes, "state decode bound exceeded"
        )

    def load_anchor_ref(self, ref: object) -> Any:
        self._tick()
        return self.codec.load_anchor_ref(ref)

    def load_child_from_delta(
        self, *, parent_state: Any, delta_ref: object, branch_from_parent: Any = None
    ) -> Any:
        self._tick()
        return self.codec.load_child_from_delta(
            parent_state=parent_state,
            delta_ref=delta_ref,
            branch_from_parent=branch_from_parent,
        )

    def dump_anchor_ref(self, state: Any) -> object:
        return self.codec.dump_anchor_ref(state)

    def dump_delta_from_parent(
        self, *, parent_state: Any, child_state: Any, branch_from_parent: Any = None
    ) -> object:
        return self.codec.dump_delta_from_parent(
            parent_state=parent_state,
            child_state=child_state,
            branch_from_parent=branch_from_parent,
        )


@dataclass
class Arena:
    """Private evaluation-only ownership; no manager/executor/runnable search."""

    tree: Tree[AlgorithmNode[Any]]
    selector: Linoo[Any]
    expansions: TreeExpansions[Any]
    random: Random
    payload: SearchRuntimeCheckpointPayload


def reconstruct(
    raw: bytes,
    codec_type: type[IncrementalStateCheckpointCodec[Any]],
    recipe: SupportedSearchRecipe,
    limits: PreparationLimits,
) -> Arena:
    """Follow native cold node/link/parent/selector restoration exactly."""
    payload = decode(raw)
    validate_source(payload, recipe, limits)
    require(
        getattr(codec_type, "prepared_domain", None) == "morpion-inline-v1",
        "codec must explicitly admit owned inline Morpion reconstruction; external stores unsupported",
    )
    codec = _BudgetCodec(codec_type(), limits)
    resolver = _create_state_resolver(
        payload=payload, state_codec=codec, state_cache_capacity=limits.max_nodes
    )
    handles = _create_state_handles(payload=payload, state_resolver=resolver)
    factory = AlgorithmNodeFactory[Any](
        tree_node_factory=TreeNodeFactory(),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        exploration_index_data_create=lambda n: create_exploration_index_data(
            n, IndexComputationType.MIN_GLOBAL_CHANGE, False
        ),
    )
    nodes = _create_nodes(
        node_factory=factory, payload=payload, state_handles_by_id=handles
    )
    _link_nodes(payload.tree.nodes, nodes_by_id=nodes)
    _restore_parent_order(
        ((n.node_id, n.parent_node_order) for n in payload.tree.nodes),
        nodes_by_id=nodes,
    )
    _restore_node_runtime_state(payload.tree.nodes, nodes_by_id=nodes)
    # Native reconstruction completes before making the bindings read-only.
    for n in nodes.values():
        n.tree_node = DetachedStructure.from_node(n.tree_node)
        n.tree_evaluation = factory.node_tree_evaluation_factory.create(n.tree_node)
        n.exploration_index_data = factory.exploration_index_data_create(n.tree_node)
    for record in payload.tree.nodes:
        _restore_evaluation(node=nodes[record.node_id], payload=record.evaluation)
        _restore_exploration_index(
            node=nodes[record.node_id], payload=record.exploration_index
        )
    tree = _build_tree(payload=payload, nodes_by_id=nodes)
    random = Random(0)
    restore_random_state(random, payload.rng_state)
    selector = Linoo[Any](
        cast("OpeningInstructor", _NoOpening()),
        random,
        depth_selection_policy=recipe.depth_selection_policy,
    )
    objective = tree.root_node.tree_evaluation.required_objective
    assert isinstance(objective, SingleAgentMaxObjective)
    assert payload.selector_state is not None
    require(
        selector.restore_from_checkpoint_payload(
            tree=tree, objective=objective, payload=payload.selector_state
        ),
        "incomplete/inconsistent Linoo checkpoint cache",
    )
    expansions = (
        TreeExpansions()
        if payload.latest_tree_expansions is None
        else _restore_tree_expansions(payload.latest_tree_expansions, nodes_by_id=nodes)
    )
    # Resolve the complete codec closure now, before any model inference.
    for n in iter_nodes(tree):
        limits.check()
        state = n.state
        require(
            getattr(state, "board", None) is not None
            and type(state.board).__module__.startswith("atomheart.games.morpion"),
            "unsupported domain state",
        )
        state_payload = cast(
            "CheckpointBackedStateHandle[Any]", handles[n.id]
        ).checkpoint_payload_for_reuse_or_none()
        assert state_payload is not None
        summary = cast("dict[str, object] | None", state_payload.state_summary)
        require(
            summary is not None
            and summary["tag"] == state.tag
            and summary["is_terminal"] == state.is_game_over(),
            "decoded state/summary identity mismatch",
        )
    return Arena(tree, selector, expansions, random, payload)


def derived(tree: Tree[Any], selector: Linoo[Any]) -> dict[str, Any]:
    """Capture native derived payloads without a second publication path."""
    context = _CheckpointBuildContext(metrics=_CheckpointBuildMetrics())
    return {
        "nodes": [
            (
                n.id,
                _build_node_evaluation_payload(n, context=context),
                _build_exploration_index_payload(n),
            )
            for n in iter_nodes(tree)
        ],
        "selector": selector.build_checkpoint_payload(
            tree.root_node.tree_evaluation.required_objective
        ),
    }


def fingerprint(
    tree: Tree[Any],
    selector: Linoo[Any],
    expansions: TreeExpansions[Any],
    version: int,
    rollout_rng_state: object,
) -> str:
    """Fingerprint stored ordered state only; never refresh or resolve owner state."""
    context = _CheckpointBuildContext(metrics=_CheckpointBuildMetrics())
    rows = []
    for n in iter_nodes(tree):
        require(
            type(n.state_handle) is CheckpointBackedStateHandle,
            "installation requires a native cold-restored owner",
        )
        rows.append(
            (
                n.id,
                n.tree_depth,
                n.all_branches_generated,
                [
                    (serialize_checkpoint_atom(b), None if child is None else child.id)
                    for b, child in n.iter_child_links()
                ],
                [
                    (parent.id, _serialize_branch_collection(branches, context=context))
                    for parent, branches in n.iter_parent_items()
                ],
                _serialize_branch_collection(
                    n.iter_unopened_branches(), context=context
                ),
                n.state_handle.checkpoint_payload_for_reuse_or_none(),
            )
        )
    return sha256(
        encode(
            (
                tree.root_node.id,
                tree.nodes_count,
                tree.branch_count,
                rows,
                derived(tree, selector),
                _build_tree_expansions_payload(expansions),
                version,
                selector.random_generator.getstate(),
                rollout_rng_state,
            )
        )
    ).hexdigest()


def preflight(
    arena: Arena, validator: Callable[[Any], bool] | None, limits: PreparationLimits
) -> None:
    """Validate complete certificate coverage before candidate evaluation."""
    for node in iter_nodes(arena.tree, reverse=True):
        limits.check()
        validate_refresh_node(node, validator)
