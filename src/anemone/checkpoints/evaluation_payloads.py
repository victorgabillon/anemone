"""Restore checkpointed node-evaluation runtime payloads."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

# ruff: noqa: TC001

from __future__ import annotations

from typing import Any, cast

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
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.utils.small_tools import Interval

from .metadata import CheckpointRestoreError
from .payloads import (
    BackupRuntimeCheckpointPayload,
    BranchFrontierCheckpointPayload,
    DecisionOrderingCheckpointPayload,
    ExplorationIndexCheckpointPayload,
    NodeEvaluationCheckpointPayload,
    PrincipalVariationCheckpointPayload,
)
from .restore_atoms import _deserialize_branch, _deserialize_optional_branch
from .value_serialization import deserialize_value


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


def _clear_private_lazy_state(node_eval: object, attribute_name: str) -> None:
    """Clear one optional lazy state slot when it exists on the evaluation."""
    if hasattr(node_eval, attribute_name):
        object.__setattr__(node_eval, attribute_name, None)


def _restore_decision_ordering(
    node_eval: object,
    payload: DecisionOrderingCheckpointPayload | None,
) -> None:
    """Restore cached branch-ordering keys when the evaluation exposes them."""
    if payload is None:
        return
    if not payload.branch_ordering:
        _clear_private_lazy_state(node_eval, "decision_ordering_")
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
    if (
        not payload.best_branch_sequence
        and payload.pv_version == 0
        and payload.cached_best_child_version is None
    ):
        _clear_private_lazy_state(node_eval, "pv_state_")
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
    if not payload.frontier_branches:
        _clear_private_lazy_state(node_eval, "branch_frontier_")
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
    if (
        payload.best_branch is None
        and payload.second_best_branch is None
        and payload.exact_child_count == 0
        and payload.selected_child_pv_version is None
        and not payload.is_initialized
    ):
        _clear_private_lazy_state(node_eval, "backup_runtime_")
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
