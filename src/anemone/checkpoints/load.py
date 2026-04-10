"""Restore live Anemone search runtimes from checkpoint payloads."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from anemone import trees
from anemone._runtime_assembly import assemble_search_runtime_dependencies
from anemone._valanga_types import AnyTurnState
from anemone.indices.node_indices.factory import (
    DepthExtendedIntervalExplo,
    DepthExtendedMinMaxPathValue,
    DepthExtendedRecurZipfQuoolExplorationData,
)
from anemone.indices.node_indices.index_data import (
    IntervalExplo,
    MaxDepthDescendants,
    MinMaxPathValue,
    NodeExplorationData,
    RecurZipfQuoolExplorationData,
)
from anemone.node_evaluation.tree.decision_ordering import BranchOrderingKey
from anemone.node_evaluation.tree.factory import NodeTreeMinmaxEvaluationFactory
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.progress_monitor.progress_monitor import create_stopping_criterion
from anemone.tree_exploration import TreeExploration
from anemone.tree_exploration_debug import (
    log_final_best_line,
    maybe_log_iteration_progress,
)
from anemone.tree_manager import TreeExpansion, TreeExpansions
from anemone.trees.descendants import RangedDescendants
from anemone.utils.small_tools import Interval

from .payloads import (
    CHECKPOINT_FORMAT_VERSION,
    AlgorithmNodeCheckpointPayload,
    BackupRuntimeCheckpointPayload,
    BranchFrontierCheckpointPayload,
    CheckpointAtomPayload,
    DecisionOrderingCheckpointPayload,
    ExplorationIndexCheckpointPayload,
    NodeEvaluationCheckpointPayload,
    PrincipalVariationCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    TreeExpansionCheckpointPayload,
    TreeExpansionsCheckpointPayload,
)
from .value_serialization import deserialize_checkpoint_atom, deserialize_value

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from valanga import BranchKey, Dynamics, RepresentationFactory, StateModifications
    from valanga.checkpoints import StateCheckpointCodec
    from valanga.evaluator_types import EvaluatorInput
    from valanga.policy import NotifyProgressCallable

    from anemone.dynamics import SearchDynamics
    from anemone.factory import SearchArgs
    from anemone.hooks.search_hooks import SearchHooks
    from anemone.node_evaluation.direct.protocols import MasterStateValueEvaluator
    from anemone.node_evaluation.tree.factory import NodeTreeEvaluationFactory


class CheckpointRestoreError(ValueError):
    """Raised when a checkpoint payload cannot be restored safely."""

    @classmethod
    def unsupported_format_version(cls, actual: int) -> CheckpointRestoreError:
        """Return the error for an unsupported checkpoint format version."""
        return cls(
            "Unsupported checkpoint format version "
            f"{actual}; expected {CHECKPOINT_FORMAT_VERSION}."
        )

    @classmethod
    def empty_tree(cls) -> CheckpointRestoreError:
        """Return the error for an empty tree checkpoint."""
        return cls("Cannot restore a checkpoint with no nodes.")

    @classmethod
    def duplicate_node_ids(cls) -> CheckpointRestoreError:
        """Return the error for duplicate node ids."""
        return cls("Checkpoint contains duplicate node ids.")

    @classmethod
    def missing_root(cls, root_node_id: int) -> CheckpointRestoreError:
        """Return the error for a root id that has no node payload."""
        return cls(f"Root node id {root_node_id} is not present in nodes.")

    @classmethod
    def invalid_exploration_index_payload(cls) -> CheckpointRestoreError:
        """Return the error for unsupported exploration-index payload shape."""
        return cls("Exploration index payloads must be dicts produced by the exporter.")

    @classmethod
    def live_tree_node_in_index_payload(cls) -> CheckpointRestoreError:
        """Return the error for accidentally serialized live tree-node pointers."""
        return cls("Exploration index payload unexpectedly contains live tree_node.")

    @classmethod
    def unsupported_exploration_index_kind(
        cls,
        kind: str | None,
    ) -> CheckpointRestoreError:
        """Return the error for an unknown exploration-index kind."""
        return cls(f"Unsupported exploration index kind {kind!r}.")

    @classmethod
    def unknown_node_id(cls, node_id: int) -> CheckpointRestoreError:
        """Return the error for a node reference that cannot be resolved."""
        return cls(f"Checkpoint references unknown node id {node_id}.")


@runtime_checkable
class _DepthCursorSelector(Protocol):
    """Selector component with a restorable depth cursor."""

    current_depth_to_expand: int


def load_search_from_checkpoint_payload[
    StateT: AnyTurnState,
    ActionT,
](
    payload: SearchRuntimeCheckpointPayload,
    *,
    state_codec: StateCheckpointCodec[StateT],
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    args: SearchArgs,
    state_type: type[StateT],
    master_state_value_evaluator: MasterStateValueEvaluator,
    random_generator: Random | None = None,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None = None,
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT] | None = None,
    notify_progress: NotifyProgressCallable | None = None,
    hooks: SearchHooks | None = None,
) -> TreeExploration[AlgorithmNode[StateT]]:
    """Restore a live search runtime from one in-memory checkpoint payload.

    The loader restores exported tree, node, value, and evaluation-cache state
    directly without evaluating checkpointed nodes. Selector-visible latest
    expansions are restored exactly from the payload. Selector-private state is
    only restored where this module has an explicit, narrow runtime contract;
    currently that means inferring Uniform's depth cursor from the latest
    expansion depth. The supplied evaluator is kept for future expansions and
    explicit reevaluations after the runtime resumes.
    """
    _validate_payload(payload)

    checkpoint_random_generator = random_generator if random_generator else Random()
    dependencies = assemble_search_runtime_dependencies(
        dynamics=dynamics,
        args=args,
        random_generator=checkpoint_random_generator,
        master_state_evaluator=master_state_value_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=(
            node_tree_evaluation_factory or NodeTreeMinmaxEvaluationFactory()
        ),
        hooks=hooks,
    )
    states_by_id: dict[int, StateT] = _load_states(
        payload=payload,
        state_codec=state_codec,
    )
    _ = state_type

    nodes_by_id = _create_nodes(
        node_factory=dependencies.tree_manager.tree_manager.node_factory,
        payload=payload,
        states_by_id=states_by_id,
    )
    _link_nodes(payload.tree.nodes, nodes_by_id=nodes_by_id)
    _restore_node_runtime_state(payload.tree.nodes, nodes_by_id=nodes_by_id)

    restored_tree = _build_tree(payload=payload, nodes_by_id=nodes_by_id)
    node_selector = dependencies.node_selector_create()
    runtime = TreeExploration(
        tree=restored_tree,
        tree_manager=dependencies.tree_manager,
        stopping_criterion=create_stopping_criterion(
            args=args.stopping_criterion,
            node_selector=node_selector,
        ),
        node_selector=node_selector,
        recommend_branch_after_exploration=args.recommender_rule,
        notify_percent_function=notify_progress,
        iteration_progress_reporter=maybe_log_iteration_progress,
        search_result_reporter=log_final_best_line,
    )
    _restore_runtime_state(
        runtime=runtime,
        payload=payload,
        random_generator=checkpoint_random_generator,
        nodes_by_id=nodes_by_id,
    )
    return runtime


def _validate_payload(payload: SearchRuntimeCheckpointPayload) -> None:
    """Validate the minimal structural facts needed before restoring."""
    if payload.format_version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointRestoreError.unsupported_format_version(payload.format_version)
    if not payload.tree.nodes:
        raise CheckpointRestoreError.empty_tree()

    node_ids = [node_payload.node_id for node_payload in payload.tree.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise CheckpointRestoreError.duplicate_node_ids()
    if payload.tree.root_node_id not in set(node_ids):
        raise CheckpointRestoreError.missing_root(payload.tree.root_node_id)


def _load_states[
    StateT: AnyTurnState,
](
    *,
    payload: SearchRuntimeCheckpointPayload,
    state_codec: StateCheckpointCodec[StateT],
) -> dict[int, StateT]:
    """Reconstruct every domain state through the external Valanga codec."""
    return {
        node_payload.node_id: state_codec.load_state_ref(node_payload.state_ref)
        for node_payload in payload.tree.nodes
    }


def _create_nodes[
    StateT: AnyTurnState,
](
    *,
    node_factory: Any,
    payload: SearchRuntimeCheckpointPayload,
    states_by_id: Mapping[int, StateT],
) -> dict[int, AlgorithmNode[StateT]]:
    """Create every node without linking parent/child edges yet.

    ``AlgorithmNodeFactory`` forwards ``count`` to ``TreeNode.id_``. Supplying
    the checkpoint id here keeps restored ids stable without relying on any
    hidden global counter.
    """
    nodes_by_id: dict[int, AlgorithmNode[StateT]] = {}
    for node_payload in sorted(payload.tree.nodes, key=lambda item: item.depth):
        node = node_factory.create(
            state=states_by_id[node_payload.node_id],
            tree_depth=node_payload.depth,
            count=node_payload.node_id,
            parent_node=None,
            branch_from_parent=None,
            modifications=None,
        )
        nodes_by_id[node_payload.node_id] = node
    return nodes_by_id


def _link_nodes(
    node_payloads: Iterable[AlgorithmNodeCheckpointPayload],
    *,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> None:
    """Restore graph edges from authoritative parent child-link payloads."""
    for node_payload in node_payloads:
        parent_node = nodes_by_id[node_payload.node_id]
        for (
            branch_payload,
            child_node_id,
        ) in node_payload.linked_children_by_branch.items():
            child_node = _require_node(nodes_by_id, child_node_id)
            branch = _deserialize_branch(branch_payload)
            parent_node.branches_children[branch] = child_node
            child_node.add_parent(branch_key=branch, new_parent_node=parent_node)


def _restore_node_runtime_state(
    node_payloads: Iterable[AlgorithmNodeCheckpointPayload],
    *,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> None:
    """Restore per-node structural flags, evaluation, and exploration index data."""
    for node_payload in node_payloads:
        node = nodes_by_id[node_payload.node_id]
        node.all_branches_generated = node_payload.generated_all_branches
        node.non_opened_branches.clear()
        node.non_opened_branches.update(
            _deserialize_branch(branch_payload)
            for branch_payload in node_payload.unopened_branches
        )
        _restore_evaluation(node=node, payload=node_payload.evaluation)
        _restore_exploration_index(node=node, payload=node_payload.exploration_index)


def _restore_evaluation(
    *,
    node: AlgorithmNode[Any],
    payload: NodeEvaluationCheckpointPayload | None,
) -> None:
    """Restore values and evaluation runtime caches for one node."""
    if payload is None:
        return

    node_eval = node.tree_evaluation
    node_eval.direct_value = (
        deserialize_value(payload.direct_value)
        if payload.direct_value is not None
        else None
    )
    node_eval.direct_evaluation_version = payload.direct_evaluation_version
    # ``backed_up_value`` is the canonical writable property over
    # ``NodeTreeEvaluationState._backed_up_value``.
    node_eval.backed_up_value = (
        deserialize_value(payload.backed_up_value)
        if payload.backed_up_value is not None
        else None
    )
    _restore_decision_ordering(node_eval, payload.decision_ordering)
    _restore_principal_variation(node_eval, payload.principal_variation)
    _restore_branch_frontier(node_eval, payload.branch_frontier)
    _restore_backup_runtime(node_eval, payload.backup_runtime)


def _restore_decision_ordering(
    node_eval: object,
    payload: DecisionOrderingCheckpointPayload | None,
) -> None:
    """Restore cached branch-ordering keys when the evaluation exposes them."""
    if payload is None:
        return
    decision_ordering = getattr(node_eval, "decision_ordering", None)
    if decision_ordering is None:
        return

    decision_ordering.branch_ordering_keys = {
        _deserialize_branch(item.branch_key): BranchOrderingKey(
            primary_score=item.primary_score,
            tactical_tiebreak=item.tactical_tiebreak,
            stable_tiebreak_id=item.stable_tiebreak_id,
        )
        for item in payload.branch_ordering
    }


def _restore_principal_variation(
    node_eval: object,
    payload: PrincipalVariationCheckpointPayload | None,
) -> None:
    """Restore principal-variation content and version metadata."""
    if payload is None:
        return
    pv_state = getattr(node_eval, "pv_state", None)
    if pv_state is None:
        return

    pv_state.best_branch_sequence = [
        _deserialize_branch(branch_payload)
        for branch_payload in payload.best_branch_sequence
    ]
    pv_state.pv_version = payload.pv_version
    pv_state.cached_best_child_version = payload.cached_best_child_version


def _restore_branch_frontier(
    node_eval: object,
    payload: BranchFrontierCheckpointPayload | None,
) -> None:
    """Restore branch-frontier membership for one node."""
    if payload is None:
        return
    branch_frontier = getattr(node_eval, "branch_frontier", None)
    if branch_frontier is None:
        return

    branch_frontier.frontier_branches = {
        _deserialize_branch(branch_payload)
        for branch_payload in payload.frontier_branches
    }


def _restore_backup_runtime(
    node_eval: object,
    payload: BackupRuntimeCheckpointPayload | None,
) -> None:
    """Restore the conservative backup-runtime cache when present."""
    if payload is None:
        return
    backup_runtime = getattr(node_eval, "backup_runtime", None)
    if backup_runtime is None:
        return

    backup_runtime.best_branch = _deserialize_optional_branch(payload.best_branch)
    backup_runtime.second_best_branch = _deserialize_optional_branch(
        payload.second_best_branch
    )
    backup_runtime.exact_child_count = payload.exact_child_count
    backup_runtime.selected_child_pv_version = payload.selected_child_pv_version
    backup_runtime.is_initialized = payload.is_initialized


def _restore_exploration_index(
    *,
    node: AlgorithmNode[Any],
    payload: ExplorationIndexCheckpointPayload | None,
) -> None:
    """Restore exploration-index data without recomputing index values."""
    if payload is None:
        node.exploration_index_data = None
        return

    index_payload = payload.payload
    if not isinstance(index_payload, dict):
        raise CheckpointRestoreError.invalid_exploration_index_payload()

    index_fields = cast("dict[str, object]", index_payload)
    index_data = _make_exploration_index_data(
        kind=payload.kind,
        node=node,
    )
    for field_name, field_value in index_fields.items():
        if field_name == "tree_node":
            raise CheckpointRestoreError.live_tree_node_in_index_payload()
        setattr(index_data, field_name, _restore_index_field_value(field_value))
    node.exploration_index_data = index_data


def _make_exploration_index_data(
    *,
    kind: str | None,
    node: AlgorithmNode[Any],
) -> NodeExplorationData[Any, Any]:
    """Create the concrete exploration-index data object named by the payload."""
    if (
        node.exploration_index_data is not None
        and type(node.exploration_index_data).__name__ == kind
    ):
        return node.exploration_index_data

    index_data_type = _EXPLORATION_INDEX_TYPES_BY_NAME.get(kind)
    if index_data_type is None:
        raise CheckpointRestoreError.unsupported_exploration_index_kind(kind)
    return index_data_type(tree_node=node.tree_node)


def _restore_index_field_value(value: object) -> object:
    """Restore one exploration-index field value."""
    if isinstance(value, dict):
        fields = cast("dict[str, object]", value)
        if set(fields) == {"min_value", "max_value"}:
            return Interval(
                min_value=cast("float | None", fields["min_value"]),
                max_value=cast("float | None", fields["max_value"]),
            )
        return fields
    return value


def _build_tree[
    StateT: AnyTurnState,
](
    *,
    payload: SearchRuntimeCheckpointPayload,
    nodes_by_id: Mapping[int, AlgorithmNode[StateT]],
) -> trees.Tree[AlgorithmNode[StateT]]:
    """Build the restored Tree and descendant registry from restored nodes."""
    root_node = nodes_by_id[payload.tree.root_node_id]
    descendants = RangedDescendants[AlgorithmNode[StateT]]()
    for node_payload in sorted(payload.tree.nodes, key=lambda item: item.depth):
        descendants.register_descendant(nodes_by_id[node_payload.node_id])

    restored_tree = trees.Tree(root_node=root_node, descendants=descendants)
    restored_tree.nodes_count = len(nodes_by_id)
    restored_tree.branch_count = sum(
        len(node_payload.linked_children_by_branch)
        for node_payload in payload.tree.nodes
    )
    return restored_tree


def _restore_runtime_state(
    *,
    runtime: TreeExploration[AlgorithmNode[Any]],
    payload: SearchRuntimeCheckpointPayload,
    random_generator: Random,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> None:
    """Restore runtime-level counters, evaluator version, RNG, and expansion log."""
    runtime.evaluator_version = payload.evaluator_version
    node_evaluator = runtime.tree_manager.node_evaluator
    if node_evaluator is not None:
        node_evaluator.current_evaluator_version = payload.evaluator_version

    if payload.rng_state is not None:
        random_generator.setstate(cast("tuple[Any, ...]", payload.rng_state))

    if payload.latest_tree_expansions is not None:
        runtime.latest_tree_expansions = _restore_tree_expansions(
            payload.latest_tree_expansions,
            nodes_by_id=nodes_by_id,
        )

    _restore_inferred_depth_selector_state(
        runtime=runtime,
        payload=payload.latest_tree_expansions,
        nodes_by_id=nodes_by_id,
    )


def _restore_tree_expansions(
    payload: TreeExpansionsCheckpointPayload,
    *,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> TreeExpansions[AlgorithmNode[Any]]:
    """Restore selector-visible expansion records from node ids."""
    tree_expansions: TreeExpansions[AlgorithmNode[Any]] = TreeExpansions()
    for expansion_payload in payload.expansions_with_node_creation:
        tree_expansions.record_creation(
            _restore_tree_expansion(expansion_payload, nodes_by_id=nodes_by_id)
        )
    for expansion_payload in payload.expansions_without_node_creation:
        tree_expansions.record_connection(
            _restore_tree_expansion(expansion_payload, nodes_by_id=nodes_by_id)
        )
    return tree_expansions


def _restore_tree_expansion(
    payload: TreeExpansionCheckpointPayload,
    *,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> TreeExpansion[AlgorithmNode[Any]]:
    """Restore one selector-visible expansion record."""
    return TreeExpansion(
        child_node=_require_node(nodes_by_id, payload.child_node_id),
        parent_node=(
            _require_node(nodes_by_id, payload.parent_node_id)
            if payload.parent_node_id is not None
            else None
        ),
        state_modifications=None,
        creation_child_node=payload.creation_child_node,
        branch_key=_deserialize_optional_branch(payload.branch_key),
    )


def _restore_inferred_depth_selector_state(
    *,
    runtime: TreeExploration[AlgorithmNode[Any]],
    payload: TreeExpansionsCheckpointPayload | None,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> None:
    """Restore simple depth selector cursors inferred from latest expansions.

    This is intentionally marked as inferred, not exact generic selector
    restoration. Uniform's only local mutable state is the next depth cursor,
    and after a completed Uniform iteration the latest touched child depth is
    exactly that next cursor. Other selector-local state should become explicit
    checkpoint payload in a future selector checkpointing PR.
    """
    if payload is None:
        return
    next_depth = _infer_latest_expansion_depth(
        payload=payload,
        nodes_by_id=nodes_by_id,
    )
    if next_depth is None:
        return

    for selector in _iter_selector_components(runtime.node_selector):
        if isinstance(selector, _DepthCursorSelector):
            selector.current_depth_to_expand = next_depth


def _infer_latest_expansion_depth(
    *,
    payload: TreeExpansionsCheckpointPayload,
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
) -> int | None:
    """Infer Uniform's next depth cursor from the latest touched child depth."""
    latest_depths = [
        _require_node(nodes_by_id, expansion.child_node_id).tree_depth
        for expansion in [
            *payload.expansions_with_node_creation,
            *payload.expansions_without_node_creation,
        ]
    ]
    if not latest_depths:
        return None
    return max(latest_depths)


def _iter_selector_components(selector: object) -> Iterable[object]:
    """Yield a composed selector and nested base selectors."""
    current: object | None = selector
    while current is not None:
        yield current
        current = getattr(current, "base", None)


def _require_node(
    nodes_by_id: Mapping[int, AlgorithmNode[Any]],
    node_id: int,
) -> AlgorithmNode[Any]:
    """Return one restored node or raise a checkpoint-specific error."""
    try:
        return nodes_by_id[node_id]
    except KeyError as exc:
        raise CheckpointRestoreError.unknown_node_id(node_id) from exc


def _deserialize_optional_branch(
    payload: CheckpointAtomPayload | None,
) -> BranchKey | None:
    """Deserialize an optional branch key payload."""
    if payload is None:
        return None
    return _deserialize_branch(payload)


def _deserialize_branch(payload: CheckpointAtomPayload) -> BranchKey:
    """Deserialize a branch key checkpoint atom."""
    return deserialize_checkpoint_atom(payload)


_EXPLORATION_INDEX_TYPES_BY_NAME: dict[
    str | None, type[NodeExplorationData[Any, Any]]
] = {
    "NodeExplorationData": NodeExplorationData,
    "RecurZipfQuoolExplorationData": RecurZipfQuoolExplorationData,
    "MinMaxPathValue": MinMaxPathValue,
    "IntervalExplo": IntervalExplo,
    "MaxDepthDescendants": MaxDepthDescendants,
    "DepthExtendedIntervalExplo": DepthExtendedIntervalExplo,
    "DepthExtendedMinMaxPathValue": DepthExtendedMinMaxPathValue,
    "DepthExtendedRecurZipfQuoolExplorationData": (
        DepthExtendedRecurZipfQuoolExplorationData
    ),
}


__all__ = [
    "CheckpointRestoreError",
    "load_search_from_checkpoint_payload",
]
