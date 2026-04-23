"""Build runtime checkpoint payloads from live Anemone searches."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from random import Random
from typing import TYPE_CHECKING, Any, cast

from anemone.utils.small_tools import Interval

from ._protocols import (
    CheckpointStateSummary,
    IncrementalStateCheckpointCodec,
    StateCheckpointSummaryCodec,
)
from .payloads import (
    AlgorithmNodeCheckpointPayload,
    AnchorCheckpointStatePayload,
    BackupRuntimeCheckpointPayload,
    BranchFrontierCheckpointPayload,
    BranchOrderingCheckpointPayload,
    CheckpointAtomPayload,
    DecisionOrderingCheckpointPayload,
    DeltaCheckpointStatePayload,
    ExplorationIndexCheckpointPayload,
    LinkedChildCheckpointPayload,
    NodeEvaluationCheckpointPayload,
    PrincipalVariationCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    SerializedValuePayload,
    TreeCheckpointPayload,
    TreeExpansionCheckpointPayload,
    TreeExpansionsCheckpointPayload,
)
from .value_serialization import serialize_checkpoint_atom, serialize_value

if TYPE_CHECKING:
    from collections.abc import Iterable, MutableMapping

    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
    from anemone.tree_exploration import TreeExploration
    from anemone.tree_manager import TreeExpansion, TreeExpansions


CHECKPOINT_ANCHOR_DEPTH_STRIDE = 4


def build_search_checkpoint_payload(
    search: TreeExploration[Any],
    *,
    state_codec: IncrementalStateCheckpointCodec[Any],
) -> SearchRuntimeCheckpointPayload:
    """Build a read-only checkpoint payload from one live search runtime.

    The new checkpoint format requires an incremental codec that can emit
    anchor snapshots plus parent-to-child deltas.
    """
    return SearchRuntimeCheckpointPayload(
        evaluator_version=search.evaluator_version,
        tree=_build_tree_payload(search=search, state_codec=state_codec),
        rng_state=_maybe_dump_rng_state(search),
        latest_tree_expansions=_build_latest_tree_expansions_payload(search),
    )


def _build_tree_payload(
    *,
    search: TreeExploration[Any],
    state_codec: IncrementalStateCheckpointCodec[Any],
) -> TreeCheckpointPayload:
    """Build the tree payload in stable depth/insertion descendant order."""
    return TreeCheckpointPayload(
        root_node_id=search.tree.root_node.id,
        nodes=[
            _build_node_payload(
                node=node,
                state_codec=state_codec,
            )
            for node in _iter_nodes_in_checkpoint_order(search)
        ],
    )


def _iter_nodes_in_checkpoint_order(
    search: TreeExploration[Any],
) -> list[AlgorithmNode[Any]]:
    """Return live nodes in stable depth order, then descendant insertion order."""
    descendants = search.tree.descendants
    return [
        node
        for tree_depth in descendants.range()
        for node in descendants[tree_depth].values()
    ]


def _build_node_payload(
    *,
    node: AlgorithmNode[Any],
    state_codec: IncrementalStateCheckpointCodec[Any],
) -> AlgorithmNodeCheckpointPayload:
    """Build one node checkpoint payload without mutating runtime state."""
    (
        parent_node,
        parent_node_id,
        branch_from_parent,
        branch_from_parent_payload,
    ) = _representative_parent_link(node)
    return AlgorithmNodeCheckpointPayload(
        node_id=node.id,
        parent_node_id=parent_node_id,
        branch_from_parent=branch_from_parent_payload,
        depth=node.tree_depth,
        state_payload=_build_checkpoint_state_payload(
            node=node,
            parent_node=parent_node,
            branch_from_parent=branch_from_parent,
            state_codec=state_codec,
        ),
        generated_all_branches=node.all_branches_generated,
        unopened_branches=_serialize_branch_collection(node.non_opened_branches),
        linked_children=_serialize_linked_children(node.branches_children),
        evaluation=_build_node_evaluation_payload(node),
        exploration_index=_build_exploration_index_payload(node),
    )


def _representative_parent_link(
    node: AlgorithmNode[Any],
) -> tuple[
    AlgorithmNode[Any] | None,
    int | None,
    BranchKey | None,
    CheckpointAtomPayload | None,
]:
    """Return a deterministic representative incoming edge for this node.

    Full incoming edge information is recoverable from every parent's
    ``linked_children`` edge list. The node-local parent fields keep the
    legacy one-parent shape by storing the first runtime parent edge.
    """
    for parent_node, branch_keys in node.parent_nodes.items():
        branch = _first_branch_in_stable_order(branch_keys)
        return (
            parent_node,
            parent_node.id,
            branch,
            serialize_checkpoint_atom(branch),
        )
    return None, None, None, None


def _build_checkpoint_state_payload(
    *,
    node: AlgorithmNode[Any],
    parent_node: AlgorithmNode[Any] | None,
    branch_from_parent: BranchKey | None,
    state_codec: IncrementalStateCheckpointCodec[Any],
) -> AnchorCheckpointStatePayload | DeltaCheckpointStatePayload:
    """Build one explicit anchor-or-delta checkpoint payload for ``node``."""
    state_summary = _dump_optional_state_summary(node.state, state_codec=state_codec)
    if _is_anchor_node(node=node, parent_node=parent_node):
        return AnchorCheckpointStatePayload(
            anchor_ref=state_codec.dump_anchor_ref(node.state),
            state_summary=state_summary,
        )

    assert parent_node is not None
    return DeltaCheckpointStatePayload(
        delta_ref=state_codec.dump_delta_from_parent(
            parent_state=parent_node.state,
            child_state=node.state,
            branch_from_parent=branch_from_parent,
        ),
        state_summary=state_summary,
    )


def _is_anchor_node(
    *,
    node: AlgorithmNode[Any],
    parent_node: AlgorithmNode[Any] | None,
) -> bool:
    """Return whether ``node`` should store a full anchor snapshot."""
    return parent_node is None or node.tree_depth % CHECKPOINT_ANCHOR_DEPTH_STRIDE == 0


def _dump_optional_state_summary[StateT: Any](
    state: StateT,
    *,
    state_codec: IncrementalStateCheckpointCodec[StateT],
) -> CheckpointStateSummary | None:
    """Dump optional checkpoint summary metadata when the codec supports it."""
    if not isinstance(state_codec, StateCheckpointSummaryCodec):
        return None
    summary_codec = cast("StateCheckpointSummaryCodec[StateT]", state_codec)
    return summary_codec.dump_state_summary(state)


def _serialize_linked_children(
    branches_children: MutableMapping[Any, AlgorithmNode[Any] | None],
) -> list[LinkedChildCheckpointPayload]:
    """Serialize linked children in stable checkpoint-atom order."""
    return sorted(
        [
            LinkedChildCheckpointPayload(
                branch_key=serialize_checkpoint_atom(branch),
                child_node_id=child.id,
            )
            for branch, child in branches_children.items()
            if child is not None
        ],
        key=lambda item: (repr(item.branch_key), item.child_node_id),
    )


def _first_branch_in_stable_order(branches: Iterable[Any]) -> Any:
    """Return the first branch under checkpoint-atom representation order."""
    ordered_branches = sorted(
        branches,
        key=lambda branch: repr(serialize_checkpoint_atom(branch)),
    )
    return ordered_branches[0]


def _build_node_evaluation_payload(
    node: AlgorithmNode[Any],
) -> NodeEvaluationCheckpointPayload:
    """Serialize the evaluation runtime state already stored on one node."""
    node_eval = node.tree_evaluation
    return NodeEvaluationCheckpointPayload(
        direct_value=_serialize_optional_value(node_eval.direct_value),
        direct_evaluation_version=node_eval.direct_evaluation_version,
        backed_up_value=_serialize_optional_value(node_eval.backed_up_value),
        decision_ordering=_build_decision_ordering_payload(node_eval),
        principal_variation=_build_principal_variation_payload(node_eval),
        branch_frontier=_build_branch_frontier_payload(node_eval),
        backup_runtime=_build_backup_runtime_payload(node_eval),
    )


def _serialize_optional_value(value: Value | None) -> SerializedValuePayload | None:
    """Serialize one optional Value payload."""
    if value is None:
        return None
    return serialize_value(value)


def _build_decision_ordering_payload(
    node_eval: Any,
) -> DecisionOrderingCheckpointPayload | None:
    """Serialize cached decision-ordering keys when the evaluation exposes them."""
    decision_ordering = getattr(node_eval, "decision_ordering", None)
    if decision_ordering is None:
        return None

    branch_ordering_keys = getattr(decision_ordering, "branch_ordering_keys", None)
    if branch_ordering_keys is None:
        return None

    return DecisionOrderingCheckpointPayload(
        branch_ordering=[
            BranchOrderingCheckpointPayload(
                branch_key=serialize_checkpoint_atom(branch),
                primary_score=ordering_key.primary_score,
                tactical_tiebreak=ordering_key.tactical_tiebreak,
                stable_tiebreak_id=ordering_key.stable_tiebreak_id,
            )
            for branch, ordering_key in branch_ordering_keys.items()
        ]
    )


def _build_principal_variation_payload(
    node_eval: Any,
) -> PrincipalVariationCheckpointPayload | None:
    """Serialize principal-variation state when present."""
    pv_state = getattr(node_eval, "pv_state", None)
    if pv_state is None:
        return None

    return PrincipalVariationCheckpointPayload(
        best_branch_sequence=[
            serialize_checkpoint_atom(branch)
            for branch in pv_state.best_branch_sequence
        ],
        pv_version=pv_state.pv_version,
        cached_best_child_version=pv_state.cached_best_child_version,
    )


def _build_branch_frontier_payload(
    node_eval: Any,
) -> BranchFrontierCheckpointPayload | None:
    """Serialize branch-frontier membership when present."""
    branch_frontier = getattr(node_eval, "branch_frontier", None)
    if branch_frontier is None:
        return None

    frontier_branches = getattr(branch_frontier, "frontier_branches", None)
    if frontier_branches is None:
        return None

    return BranchFrontierCheckpointPayload(
        frontier_branches=_serialize_branch_collection(frontier_branches)
    )


def _build_backup_runtime_payload(
    node_eval: Any,
) -> BackupRuntimeCheckpointPayload | None:
    """Serialize conservative backup-runtime cache state when present."""
    backup_runtime = getattr(node_eval, "backup_runtime", None)
    if backup_runtime is None:
        return None

    return BackupRuntimeCheckpointPayload(
        best_branch=_serialize_optional_atom(backup_runtime.best_branch),
        second_best_branch=_serialize_optional_atom(backup_runtime.second_best_branch),
        exact_child_count=backup_runtime.exact_child_count,
        selected_child_pv_version=backup_runtime.selected_child_pv_version,
        is_initialized=backup_runtime.is_initialized,
    )


def _serialize_branch_collection(
    branches: Iterable[Any],
) -> list[CheckpointAtomPayload]:
    """Serialize a branch collection in deterministic checkpoint-atom order."""
    return sorted(
        (serialize_checkpoint_atom(branch) for branch in branches),
        key=repr,
    )


def _serialize_optional_atom(value: object | None) -> CheckpointAtomPayload | None:
    """Serialize a small optional atom payload."""
    if value is None:
        return None
    return serialize_checkpoint_atom(value)


def _build_exploration_index_payload(
    node: AlgorithmNode[Any],
) -> ExplorationIndexCheckpointPayload | None:
    """Serialize stable, pointer-free exploration-index fields when present."""
    index_data = node.exploration_index_data
    if index_data is None:
        return None

    return ExplorationIndexCheckpointPayload(
        kind=type(index_data).__name__,
        payload=_exploration_index_fields(index_data),
    )


def _exploration_index_fields(index_data: object) -> dict[str, object]:
    """Return dataclass index fields while excluding live tree-node pointers."""
    if not is_dataclass(index_data):
        return {"index": getattr(index_data, "index", None)}

    payload: dict[str, object] = {}
    for field in fields(index_data):
        if field.name == "tree_node":
            continue
        value = getattr(index_data, field.name)
        payload[field.name] = _serialize_index_field_value(value)
    return payload


def _serialize_index_field_value(value: object) -> object:
    """Serialize one exploration-index field into a pointer-free payload."""
    if isinstance(value, Interval):
        return {
            "min_value": value.min_value,
            "max_value": value.max_value,
        }
    return value


def _maybe_dump_rng_state(search: TreeExploration[Any]) -> object | None:
    """Return directly exposed selector RNG state when available."""
    random_generator = getattr(search.node_selector, "random_generator", None)
    if isinstance(random_generator, Random):
        return random_generator.getstate()

    # TreeExploration does not retain the explore() RNG. Selector-specific RNG
    # restoration can be made more complete with selector checkpoint payloads.
    return None


def _build_latest_tree_expansions_payload(
    search: TreeExploration[Any],
) -> TreeExpansionsCheckpointPayload:
    """Serialize selector-visible expansion records from the latest iteration."""
    return _build_tree_expansions_payload(search.latest_tree_expansions)


def _build_tree_expansions_payload(
    tree_expansions: TreeExpansions[Any],
) -> TreeExpansionsCheckpointPayload:
    """Serialize one TreeExpansions object by node ids and branch keys."""
    return TreeExpansionsCheckpointPayload(
        expansions_with_node_creation=[
            _build_tree_expansion_payload(tree_expansion)
            for tree_expansion in tree_expansions.expansions_with_node_creation
        ],
        expansions_without_node_creation=[
            _build_tree_expansion_payload(tree_expansion)
            for tree_expansion in tree_expansions.expansions_without_node_creation
        ],
    )


def _build_tree_expansion_payload(
    tree_expansion: TreeExpansion[Any],
) -> TreeExpansionCheckpointPayload:
    """Serialize one TreeExpansion object without state reconstruction data."""
    return TreeExpansionCheckpointPayload(
        child_node_id=tree_expansion.child_node.id,
        parent_node_id=(
            tree_expansion.parent_node.id
            if tree_expansion.parent_node is not None
            else None
        ),
        branch_key=_serialize_optional_atom(tree_expansion.branch_key),
        creation_child_node=tree_expansion.creation_child_node,
    )


__all__ = ["build_search_checkpoint_payload"]
