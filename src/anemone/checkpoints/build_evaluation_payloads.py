"""Build checkpoint payloads for node evaluation runtime state."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

# ruff: noqa: TC001

from __future__ import annotations

from dataclasses import fields, is_dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone.utils.small_tools import Interval

from .build_atoms import (
    _serialize_branch_collection,
    _serialize_evaluation_atom_for_build,
    _serialize_optional_evaluation_atom,
)
from .build_context import _CheckpointBuildContext
from .build_values import _serialize_optional_value
from .payloads import (
    BackupRuntimeCheckpointPayload,
    BranchFrontierCheckpointPayload,
    BranchOrderingCheckpointPayload,
    DecisionOrderingCheckpointPayload,
    ExplorationIndexCheckpointPayload,
    NodeEvaluationCheckpointPayload,
    PrincipalVariationCheckpointPayload,
)

if TYPE_CHECKING:
    from anemone.node_evaluation.common.branch_frontier import BranchFrontierState
    from anemone.node_evaluation.common.principal_variation import (
        PrincipalVariationState,
    )
    from anemone.node_evaluation.tree.decision_ordering import DecisionOrderingState
    from anemone.node_evaluation.tree.top2_exactness_pv_runtime import (
        Top2ExactnessPvRuntime,
    )
    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode


def _build_node_evaluation_payload(
    node: AlgorithmNode[Any],
    *,
    context: _CheckpointBuildContext,
) -> NodeEvaluationCheckpointPayload:
    """Serialize the evaluation runtime state already stored on one node."""
    context.metrics.node_evaluation_calls += 1
    started_at = perf_counter()
    try:
        tree_eval_access_started_at = perf_counter()
        node_eval = node.tree_evaluation
        context.metrics.tree_evaluation_access_calls += 1
        context.metrics.tree_evaluation_access_s += (
            perf_counter() - tree_eval_access_started_at
        )

        direct_value_access_started_at = perf_counter()
        direct_value = node_eval.direct_value
        context.metrics.direct_value_access_calls += 1
        context.metrics.direct_value_access_s += (
            perf_counter() - direct_value_access_started_at
        )

        backed_up_value_access_started_at = perf_counter()
        backed_up_value = node_eval.backed_up_value
        context.metrics.backed_up_value_access_calls += 1
        context.metrics.backed_up_value_access_s += (
            perf_counter() - backed_up_value_access_started_at
        )

        return NodeEvaluationCheckpointPayload(
            direct_value=_serialize_optional_value(
                direct_value,
                context=context,
                value_kind="direct",
            ),
            direct_evaluation_version=node_eval.direct_evaluation_version,
            backed_up_value=_serialize_optional_value(
                backed_up_value,
                context=context,
                value_kind="backed_up",
            ),
            decision_ordering=_build_decision_ordering_payload(
                node_eval,
                context=context,
            ),
            principal_variation=_build_principal_variation_payload(
                node_eval,
                context=context,
            ),
            branch_frontier=_build_branch_frontier_payload(
                node_eval,
                context=context,
            ),
            backup_runtime=_build_backup_runtime_payload(
                node_eval,
                context=context,
            ),
        )
    finally:
        context.metrics.node_evaluation_total_s += perf_counter() - started_at


def _build_decision_ordering_payload(
    node_eval: Any,
    *,
    context: _CheckpointBuildContext,
) -> DecisionOrderingCheckpointPayload | None:
    """Serialize cached decision-ordering keys when the evaluation exposes them."""
    state_getter = getattr(node_eval, "decision_ordering_state_or_none", None)
    if callable(state_getter):
        decision_ordering = cast("DecisionOrderingState | None", state_getter())
        if decision_ordering is None:
            return DecisionOrderingCheckpointPayload()
    else:
        decision_ordering = cast(
            "DecisionOrderingState | None",
            getattr(node_eval, "decision_ordering", None),
        )
    if decision_ordering is None:
        return None

    branch_ordering_keys = getattr(decision_ordering, "branch_ordering_keys", None)
    if branch_ordering_keys is None:
        return None

    context.metrics.decision_ordering_calls += 1
    started_at = perf_counter()
    try:
        return DecisionOrderingCheckpointPayload(
            branch_ordering=[
                BranchOrderingCheckpointPayload(
                    branch_key=_serialize_evaluation_atom_for_build(
                        branch,
                        context=context,
                    ),
                    primary_score=ordering_key.primary_score,
                    tactical_tiebreak=ordering_key.tactical_tiebreak,
                    stable_tiebreak_id=ordering_key.stable_tiebreak_id,
                )
                for branch, ordering_key in branch_ordering_keys.items()
            ]
        )
    finally:
        context.metrics.decision_ordering_serialize_s += perf_counter() - started_at


def _build_principal_variation_payload(
    node_eval: Any,
    *,
    context: _CheckpointBuildContext,
) -> PrincipalVariationCheckpointPayload | None:
    """Serialize principal-variation state when present."""
    state_getter = getattr(node_eval, "pv_state_or_none", None)
    if callable(state_getter):
        pv_state = cast("PrincipalVariationState | None", state_getter())
        if pv_state is None:
            return PrincipalVariationCheckpointPayload()
    else:
        pv_state = cast(
            "PrincipalVariationState | None",
            getattr(node_eval, "pv_state", None),
        )
    if pv_state is None:
        return None

    context.metrics.principal_variation_calls += 1
    started_at = perf_counter()
    try:
        return PrincipalVariationCheckpointPayload(
            best_branch_sequence=[
                _serialize_evaluation_atom_for_build(branch, context=context)
                for branch in pv_state.best_branch_sequence
            ],
            pv_version=pv_state.pv_version,
            cached_best_child_version=pv_state.cached_best_child_version,
        )
    finally:
        context.metrics.principal_variation_serialize_s += perf_counter() - started_at


def _build_branch_frontier_payload(
    node_eval: Any,
    *,
    context: _CheckpointBuildContext,
) -> BranchFrontierCheckpointPayload | None:
    """Serialize branch-frontier membership when present."""
    state_getter = getattr(node_eval, "branch_frontier_state_or_none", None)
    if callable(state_getter):
        branch_frontier = cast("BranchFrontierState | None", state_getter())
        if branch_frontier is None:
            return BranchFrontierCheckpointPayload()
    else:
        branch_frontier = cast(
            "BranchFrontierState | None",
            getattr(node_eval, "branch_frontier", None),
        )
    if branch_frontier is None:
        return None

    frontier_branches = getattr(branch_frontier, "frontier_branches", None)
    if frontier_branches is None:
        return None

    context.metrics.branch_frontier_calls += 1
    started_at = perf_counter()
    try:
        return BranchFrontierCheckpointPayload(
            frontier_branches=_serialize_branch_collection(
                frontier_branches,
                context=context,
                atom_scope="evaluation",
            )
        )
    finally:
        context.metrics.branch_frontier_total_s += perf_counter() - started_at


def _build_backup_runtime_payload(
    node_eval: Any,
    *,
    context: _CheckpointBuildContext,
) -> BackupRuntimeCheckpointPayload | None:
    """Serialize conservative backup-runtime cache state when present."""
    state_getter = getattr(node_eval, "backup_runtime_state_or_none", None)
    if callable(state_getter):
        backup_runtime = cast("Top2ExactnessPvRuntime[Any] | None", state_getter())
        if backup_runtime is None:
            return BackupRuntimeCheckpointPayload()
    else:
        backup_runtime = cast(
            "Top2ExactnessPvRuntime[Any] | None",
            getattr(node_eval, "backup_runtime", None),
        )
    if backup_runtime is None:
        return None

    context.metrics.backup_runtime_calls += 1
    started_at = perf_counter()
    try:
        return BackupRuntimeCheckpointPayload(
            best_branch=_serialize_optional_evaluation_atom(
                backup_runtime.best_branch,
                context=context,
            ),
            second_best_branch=_serialize_optional_evaluation_atom(
                backup_runtime.second_best_branch,
                context=context,
            ),
            exact_child_count=backup_runtime.exact_child_count,
            selected_child_pv_version=backup_runtime.selected_child_pv_version,
            is_initialized=backup_runtime.is_initialized,
        )
    finally:
        context.metrics.backup_runtime_total_s += perf_counter() - started_at


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
    for dataclass_field in fields(index_data):
        if dataclass_field.name == "tree_node":
            continue
        value = getattr(index_data, dataclass_field.name)
        payload[dataclass_field.name] = _serialize_index_field_value(value)
    return payload


def _serialize_index_field_value(value: object) -> object:
    """Serialize one exploration-index field into a pointer-free payload."""
    if isinstance(value, Interval):
        return {
            "min_value": value.min_value,
            "max_value": value.max_value,
        }
    return value
