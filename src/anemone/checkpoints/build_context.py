"""Build-local checkpoint context, metrics, and profiling helpers."""

# pyright: reportUnusedClass=false, reportUnusedFunction=false

# ruff: noqa: TC001

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from anemone.utils.logger import anemone_logger

from .payloads import CheckpointAtomPayload, SerializedValuePayload

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

CHECKPOINT_ANCHOR_DEPTH_STRIDE = 4
CHECKPOINT_ANCHOR_FALLBACK_WARNING_FRACTION = 0.10


def _empty_atom_serialization_cache() -> dict[object, CheckpointAtomPayload]:
    """Return an empty typed atom-serialization cache."""
    return {}


def _empty_value_identity_serialization_cache() -> dict[
    int, tuple[object, SerializedValuePayload]
]:
    """Return an empty typed value-serialization cache."""
    return {}


@dataclass(slots=True)
class _CheckpointBuildMetrics:
    """Aggregate metrics for one checkpoint build."""

    delta_candidates_attempted: int = 0
    delta_candidates_rejected: int = 0
    state_payloads_reused: int = 0
    anchor_payloads_reused: int = 0
    delta_payloads_reused: int = 0
    state_payload_reuse_rejected: int = 0
    delta_payloads_emitted: int = 0
    anchor_fallbacks: int = 0
    node_count: int = 0
    anchor_payloads_emitted: int = 0
    iter_nodes_s: float = 0.0
    tree_payload_s: float = 0.0
    node_payload_total_s: float = 0.0
    state_payload_total_s: float = 0.0
    state_payload_reuse_s: float = 0.0
    anchor_state_total_s: float = 0.0
    delta_state_total_s: float = 0.0
    state_summary_total_s: float = 0.0
    evaluation_payload_total_s: float = 0.0
    exploration_index_total_s: float = 0.0
    linked_children_total_s: float = 0.0
    unopened_branches_total_s: float = 0.0
    latest_expansions_total_s: float = 0.0
    selector_state_total_s: float = 0.0
    rng_state_total_s: float = 0.0
    node_evaluation_calls: int = 0
    node_evaluation_total_s: float = 0.0
    evaluation_payload_reuse_candidates_missing_version: int = 0
    evaluation_payload_reuse_blocked_missing_version: int = 0
    tree_evaluation_access_calls: int = 0
    tree_evaluation_access_s: float = 0.0
    direct_value_access_calls: int = 0
    direct_value_access_s: float = 0.0
    direct_value_serialize_calls: int = 0
    direct_value_serialize_s: float = 0.0
    backed_up_value_access_calls: int = 0
    backed_up_value_access_s: float = 0.0
    backed_up_value_serialize_calls: int = 0
    backed_up_value_serialize_s: float = 0.0
    serialize_value_calls: int = 0
    serialize_value_total_s: float = 0.0
    serialize_value_cache_hits: int = 0
    serialize_value_cache_misses: int = 0
    value_semantic_validation_calls: int = 0
    value_semantic_validation_s: float = 0.0
    principal_variation_calls: int = 0
    principal_variation_serialize_s: float = 0.0
    decision_ordering_calls: int = 0
    decision_ordering_serialize_s: float = 0.0
    branch_frontier_calls: int = 0
    branch_frontier_total_s: float = 0.0
    backup_runtime_calls: int = 0
    backup_runtime_total_s: float = 0.0
    branch_collection_calls: int = 0
    branch_collection_total_s: float = 0.0
    branch_collection_sort_calls: int = 0
    branch_collection_sort_s: float = 0.0
    representative_parent_calls: int = 0
    representative_parent_total_s: float = 0.0
    reusable_parent_scan_calls: int = 0
    reusable_parent_scan_s: float = 0.0
    stored_or_first_branch_calls: int = 0
    stored_or_first_branch_total_s: float = 0.0
    first_branch_calls: int = 0
    first_branch_total_s: float = 0.0
    linked_children_calls: int = 0
    linked_children_sort_calls: int = 0
    linked_children_sort_s: float = 0.0
    atom_serialize_calls: int = 0
    atom_serialize_cache_hits: int = 0
    atom_serialize_cache_misses: int = 0
    atom_serialize_total_s: float = 0.0
    evaluation_atom_serialize_calls: int = 0
    evaluation_atom_serialize_total_s: float = 0.0
    evaluation_atom_serialize_cache_hits: int = 0
    evaluation_atom_serialize_cache_misses: int = 0


@dataclass(slots=True)
class _CheckpointBuildContext:
    """Build-local caches used to avoid repeated deterministic work."""

    metrics: _CheckpointBuildMetrics
    atom_serialization_cache: dict[object, CheckpointAtomPayload] = field(
        default_factory=_empty_atom_serialization_cache
    )
    value_identity_serialization_cache: dict[
        int, tuple[object, SerializedValuePayload]
    ] = field(default_factory=_empty_value_identity_serialization_cache)


@dataclass(frozen=True, slots=True)
class _ParentBranchSerialization:
    """Stable serialized branch ordering for one parent edge set."""

    parent_node: AlgorithmNode[Any]
    ordered_branches: tuple[BranchKey, ...]
    serialized_branches: tuple[CheckpointAtomPayload, ...]


@dataclass(slots=True)
class _NodeCheckpointBuildCache:
    """Per-node cached parent branch ordering reused across helpers."""

    parent_branch_entries: list[_ParentBranchSerialization]


def _log_checkpoint_build_metrics(metrics: _CheckpointBuildMetrics) -> None:
    """Emit aggregate checkpoint-build metrics once per build."""
    anemone_logger.info(
        "[checkpoint-metrics] delta_attempts=%s delta_rejected=%s "
        "delta_emitted=%s anchor_fallbacks=%s state_payloads_reused=%s",
        metrics.delta_candidates_attempted,
        metrics.delta_candidates_rejected,
        metrics.delta_payloads_emitted,
        metrics.anchor_fallbacks,
        metrics.state_payloads_reused,
    )
    attempted_delta_nodes = metrics.delta_payloads_emitted + metrics.anchor_fallbacks
    if attempted_delta_nodes == 0:
        return
    if (
        metrics.anchor_fallbacks / attempted_delta_nodes
        > CHECKPOINT_ANCHOR_FALLBACK_WARNING_FRACTION
    ):
        anemone_logger.warning(
            "Checkpoint anchor fallback rate is high: anchor_fallbacks=%s "
            "delta_nodes=%s. This may indicate representative-parent/state-parent mismatch inefficiency.",
            metrics.anchor_fallbacks,
            attempted_delta_nodes,
        )
    anemone_logger.info(
        "[checkpoint-profile] node_count=%s anchors=%s deltas=%s "
        "delta_attempts=%s delta_rejected=%s anchor_fallbacks=%s "
        "state_payloads_reused=%s anchor_payloads_reused=%s "
        "delta_payloads_reused=%s state_payload_reuse_rejected=%s "
        "iter_nodes_s=%.6f tree_payload_s=%.6f node_payload_total_s=%.6f "
        "state_payload_total_s=%.6f state_payload_reuse_s=%.6f anchor_state_total_s=%.6f "
        "delta_state_total_s=%.6f state_summary_total_s=%.6f "
        "evaluation_payload_total_s=%.6f exploration_index_total_s=%.6f "
        "linked_children_total_s=%.6f unopened_branches_total_s=%.6f "
        "latest_expansions_total_s=%.6f selector_state_total_s=%.6f "
        "rng_state_total_s=%.6f",
        metrics.node_count,
        metrics.anchor_payloads_emitted,
        metrics.delta_payloads_emitted,
        metrics.delta_candidates_attempted,
        metrics.delta_candidates_rejected,
        metrics.anchor_fallbacks,
        metrics.state_payloads_reused,
        metrics.anchor_payloads_reused,
        metrics.delta_payloads_reused,
        metrics.state_payload_reuse_rejected,
        metrics.iter_nodes_s,
        metrics.tree_payload_s,
        metrics.node_payload_total_s,
        metrics.state_payload_total_s,
        metrics.state_payload_reuse_s,
        metrics.anchor_state_total_s,
        metrics.delta_state_total_s,
        metrics.state_summary_total_s,
        metrics.evaluation_payload_total_s,
        metrics.exploration_index_total_s,
        metrics.linked_children_total_s,
        metrics.unopened_branches_total_s,
        metrics.latest_expansions_total_s,
        metrics.selector_state_total_s,
        metrics.rng_state_total_s,
    )
    anemone_logger.info(
        "[checkpoint-profile-rates] node_avg_ms=%.6f anchor_avg_ms=%.6f "
        "delta_avg_ms=%.6f summary_avg_ms=%.6f",
        _average_ms(metrics.node_payload_total_s, metrics.node_count),
        _average_ms(metrics.anchor_state_total_s, metrics.anchor_payloads_emitted),
        _average_ms(metrics.delta_state_total_s, metrics.delta_payloads_emitted),
        _average_ms(metrics.state_summary_total_s, metrics.node_count),
    )
    anemone_logger.info(
        "[checkpoint-build-detail] node_evaluation_calls=%s node_evaluation_total_s=%.6f "
        "evaluation_payload_reuse_candidates_missing_version=%s "
        "evaluation_payload_reuse_blocked_missing_version=%s "
        "tree_evaluation_access_calls=%s tree_evaluation_access_s=%.6f "
        "direct_value_access_calls=%s direct_value_access_s=%.6f "
        "direct_value_serialize_calls=%s direct_value_serialize_s=%.6f "
        "backed_up_value_access_calls=%s backed_up_value_access_s=%.6f "
        "backed_up_value_serialize_calls=%s backed_up_value_serialize_s=%.6f "
        "serialize_value_calls=%s serialize_value_total_s=%.6f "
        "serialize_value_cache_hits=%s serialize_value_cache_misses=%s "
        "value_semantic_validation_calls=%s value_semantic_validation_s=%.6f "
        "pv_serialize_calls=%s pv_serialize_s=%.6f "
        "decision_ordering_calls=%s decision_ordering_serialize_s=%.6f "
        "branch_frontier_calls=%s branch_frontier_total_s=%.6f "
        "backup_runtime_calls=%s backup_runtime_total_s=%.6f "
        "branch_collection_calls=%s branch_collection_total_s=%.6f "
        "branch_collection_sort_calls=%s branch_collection_sort_s=%.6f "
        "representative_parent_calls=%s representative_parent_total_s=%.6f "
        "reusable_parent_scan_calls=%s reusable_parent_scan_s=%.6f "
        "stored_or_first_branch_calls=%s stored_or_first_branch_total_s=%.6f "
        "first_branch_calls=%s first_branch_total_s=%.6f "
        "linked_children_calls=%s linked_children_total_s=%.6f "
        "linked_children_sort_calls=%s linked_children_sort_s=%.6f "
        "evaluation_atom_serialize_calls=%s evaluation_atom_serialize_total_s=%.6f "
        "evaluation_atom_serialize_cache_hits=%s evaluation_atom_serialize_cache_misses=%s "
        "atom_serialize_calls=%s atom_serialize_total_s=%.6f "
        "atom_serialize_cache_hits=%s atom_serialize_cache_misses=%s",
        metrics.node_evaluation_calls,
        metrics.node_evaluation_total_s,
        metrics.evaluation_payload_reuse_candidates_missing_version,
        metrics.evaluation_payload_reuse_blocked_missing_version,
        metrics.tree_evaluation_access_calls,
        metrics.tree_evaluation_access_s,
        metrics.direct_value_access_calls,
        metrics.direct_value_access_s,
        metrics.direct_value_serialize_calls,
        metrics.direct_value_serialize_s,
        metrics.backed_up_value_access_calls,
        metrics.backed_up_value_access_s,
        metrics.backed_up_value_serialize_calls,
        metrics.backed_up_value_serialize_s,
        metrics.serialize_value_calls,
        metrics.serialize_value_total_s,
        metrics.serialize_value_cache_hits,
        metrics.serialize_value_cache_misses,
        metrics.value_semantic_validation_calls,
        metrics.value_semantic_validation_s,
        metrics.principal_variation_calls,
        metrics.principal_variation_serialize_s,
        metrics.decision_ordering_calls,
        metrics.decision_ordering_serialize_s,
        metrics.branch_frontier_calls,
        metrics.branch_frontier_total_s,
        metrics.backup_runtime_calls,
        metrics.backup_runtime_total_s,
        metrics.branch_collection_calls,
        metrics.branch_collection_total_s,
        metrics.branch_collection_sort_calls,
        metrics.branch_collection_sort_s,
        metrics.representative_parent_calls,
        metrics.representative_parent_total_s,
        metrics.reusable_parent_scan_calls,
        metrics.reusable_parent_scan_s,
        metrics.stored_or_first_branch_calls,
        metrics.stored_or_first_branch_total_s,
        metrics.first_branch_calls,
        metrics.first_branch_total_s,
        metrics.linked_children_calls,
        metrics.linked_children_total_s,
        metrics.linked_children_sort_calls,
        metrics.linked_children_sort_s,
        metrics.evaluation_atom_serialize_calls,
        metrics.evaluation_atom_serialize_total_s,
        metrics.evaluation_atom_serialize_cache_hits,
        metrics.evaluation_atom_serialize_cache_misses,
        metrics.atom_serialize_calls,
        metrics.atom_serialize_total_s,
        metrics.atom_serialize_cache_hits,
        metrics.atom_serialize_cache_misses,
    )


def _average_ms(total_s: float, count: int) -> float:
    """Return a stable milliseconds average with zero-safe division."""
    if count <= 0:
        return 0.0
    return 1000.0 * total_s / count


def _maybe_log_checkpoint_codec_profile(state_codec: object) -> None:
    """Log optional codec-provided aggregate profiling when available."""
    profile_snapshot = getattr(state_codec, "checkpoint_profile_snapshot", None)
    if not callable(profile_snapshot):
        return
    snapshot = profile_snapshot()
    if not isinstance(snapshot, Mapping):
        return
    profile_mapping = cast("Mapping[object, object]", snapshot)
    anemone_logger.info(
        "[checkpoint-codec-profile] %s",
        _format_profile_mapping(profile_mapping),
    )


def _maybe_reset_checkpoint_profile(state_codec: object) -> None:
    """Reset optional codec-provided aggregate profiling when available."""
    reset_profile = getattr(state_codec, "reset_checkpoint_profile", None)
    if callable(reset_profile):
        reset_profile()


def _format_profile_mapping(profile: Mapping[object, object]) -> str:
    """Format a small codec profile mapping as stable key=value pairs."""
    items = [
        f"{key}={_format_profile_value(profile[key])}"
        for key in sorted(profile, key=str)
    ]
    return " ".join(items)


def _format_profile_value(value: object) -> str:
    """Format one codec profile value for structured logs."""
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
