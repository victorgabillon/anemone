"""Build runtime checkpoint payloads from live Anemone searches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from random import Random
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from anemone.node_evaluation.common import canonical_value
from anemone.node_selector import StatefulNodeSelector
from anemone.objectives import SingleAgentMaxObjective
from anemone.utils.logger import anemone_logger, checkpoint_logger
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
    SelectorCheckpointPayload,
    SerializedOverEventPayload,
    SerializedValuePayload,
    TreeCheckpointPayload,
    TreeExpansionCheckpointPayload,
    TreeExpansionsCheckpointPayload,
)
from .state_handles import checkpoint_payload_for_reuse_or_none
from .value_serialization import (
    serialize_checkpoint_atom,
    serialize_over_event_with,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
    from anemone.tree_exploration import TreeExploration
    from anemone.tree_manager import TreeExpansion, TreeExpansions


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


def build_search_checkpoint_payload(
    search: TreeExploration[Any],
    *,
    state_codec: IncrementalStateCheckpointCodec[Any],
) -> SearchRuntimeCheckpointPayload:
    """Build a read-only checkpoint payload from one live search runtime.

    The new checkpoint format requires an incremental codec that can emit
    anchor snapshots plus parent-to-child deltas.
    """
    metrics = _CheckpointBuildMetrics()
    context = _CheckpointBuildContext(metrics=metrics)
    _maybe_reset_checkpoint_profile(state_codec)
    tree_payload_started_at = perf_counter()
    tree_payload = _build_tree_payload(
        search=search,
        state_codec=state_codec,
        metrics=metrics,
        context=context,
    )
    metrics.tree_payload_s += perf_counter() - tree_payload_started_at
    rng_state_started_at = perf_counter()
    rng_state = _maybe_dump_rng_state(search)
    metrics.rng_state_total_s += perf_counter() - rng_state_started_at
    latest_tree_expansions_started_at = perf_counter()
    latest_tree_expansions = _build_latest_tree_expansions_payload(search)
    metrics.latest_expansions_total_s += (
        perf_counter() - latest_tree_expansions_started_at
    )
    selector_state_started_at = perf_counter()
    selector_state = _build_selector_state_payload(search)
    metrics.selector_state_total_s += perf_counter() - selector_state_started_at
    _log_checkpoint_build_metrics(metrics)
    _maybe_log_checkpoint_codec_profile(state_codec)
    _maybe_reset_checkpoint_profile(state_codec)
    return SearchRuntimeCheckpointPayload(
        evaluator_version=search.evaluator_version,
        tree=tree_payload,
        rng_state=rng_state,
        latest_tree_expansions=latest_tree_expansions,
        selector_state=selector_state,
    )


def _build_tree_payload(
    *,
    search: TreeExploration[Any],
    state_codec: IncrementalStateCheckpointCodec[Any],
    metrics: _CheckpointBuildMetrics,
    context: _CheckpointBuildContext,
) -> TreeCheckpointPayload:
    """Build the tree payload in stable depth/insertion descendant order."""
    iter_nodes_started_at = perf_counter()
    nodes = _iter_nodes_in_checkpoint_order(search)
    metrics.iter_nodes_s += perf_counter() - iter_nodes_started_at
    metrics.node_count = len(nodes)
    payload_nodes: list[AlgorithmNodeCheckpointPayload] = []
    for node in nodes:
        node_payload_started_at = perf_counter()
        payload_nodes.append(
            _build_node_payload(
                node=node,
                state_codec=state_codec,
                metrics=metrics,
                context=context,
            )
        )
        metrics.node_payload_total_s += perf_counter() - node_payload_started_at
    return TreeCheckpointPayload(
        root_node_id=search.tree.root_node.id,
        nodes=payload_nodes,
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
    metrics: _CheckpointBuildMetrics,
    context: _CheckpointBuildContext,
) -> AlgorithmNodeCheckpointPayload:
    """Build one node checkpoint payload without mutating runtime state."""
    # Safe cross-generation evaluation payload reuse needs one explicit runtime
    # version that changes whenever any serialized evaluation field changes.
    # Until that exists, restored/checkpoint-backed nodes are only counted as
    # reuse candidates and are always rebuilt to avoid stale payload reuse after
    # reevaluation patches, backup propagation, or frontier/order updates.
    if checkpoint_payload_for_reuse_or_none(node.state_handle) is not None:
        metrics.evaluation_payload_reuse_candidates_missing_version += 1
        metrics.evaluation_payload_reuse_blocked_missing_version += 1

    node_cache = _build_node_checkpoint_cache(node, context=context)
    representative_parent_started_at = perf_counter()
    (
        representative_parent,
        parent_node_id,
        _branch_from_parent,
        branch_from_parent_payload,
    ) = _representative_parent_link(
        node,
        node_cache=node_cache,
        context=context,
    )
    metrics.representative_parent_calls += 1
    metrics.representative_parent_total_s += (
        perf_counter() - representative_parent_started_at
    )
    unopened_branches_started_at = perf_counter()
    unopened_branches = _serialize_branch_collection(
        node.iter_unopened_branches(),
        context=context,
    )
    metrics.unopened_branches_total_s += perf_counter() - unopened_branches_started_at
    linked_children_started_at = perf_counter()
    linked_children = _serialize_linked_children(
        node.iter_child_links(),
        context=context,
    )
    metrics.linked_children_total_s += perf_counter() - linked_children_started_at
    evaluation_started_at = perf_counter()
    evaluation = _build_node_evaluation_payload(node, context=context)
    metrics.evaluation_payload_total_s += perf_counter() - evaluation_started_at
    exploration_index_started_at = perf_counter()
    exploration_index = _build_exploration_index_payload(node)
    metrics.exploration_index_total_s += perf_counter() - exploration_index_started_at
    return AlgorithmNodeCheckpointPayload(
        node_id=node.id,
        parent_node_id=parent_node_id,
        branch_from_parent=branch_from_parent_payload,
        depth=node.tree_depth,
        state_payload=_build_checkpoint_state_payload(
            node=node,
            representative_parent=representative_parent,
            state_codec=state_codec,
            metrics=metrics,
            context=context,
            node_cache=node_cache,
        ),
        generated_all_branches=node.all_branches_generated,
        unopened_branches=unopened_branches,
        linked_children=linked_children,
        evaluation=evaluation,
        exploration_index=exploration_index,
    )


def _build_node_checkpoint_cache(
    node: AlgorithmNode[Any],
    *,
    context: _CheckpointBuildContext,
) -> _NodeCheckpointBuildCache:
    """Precompute parent branch ordering reused across node helper calls."""
    return _NodeCheckpointBuildCache(
        parent_branch_entries=[
            _serialize_parent_branches(
                parent_node,
                branch_keys,
                context=context,
            )
            for parent_node, branch_keys in node.parent_nodes.items()
        ]
    )


def _representative_parent_link(
    node: AlgorithmNode[Any],
    *,
    node_cache: _NodeCheckpointBuildCache,
    context: _CheckpointBuildContext,
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
    preferred_parent_link = _preferred_parent_link_from_reusable_state_payload(
        node,
        node_cache=node_cache,
        context=context,
    )
    if preferred_parent_link is not None:
        return preferred_parent_link

    for parent_branch_entry in node_cache.parent_branch_entries:
        branch = _first_branch_in_stable_order(
            parent_branch_entry,
            context=context,
        )
        return (
            parent_branch_entry.parent_node,
            parent_branch_entry.parent_node.id,
            branch,
            _serialize_checkpoint_atom_for_build(branch, context=context),
        )
    return None, None, None, None


def _preferred_parent_link_from_reusable_state_payload(
    node: AlgorithmNode[Any],
    *,
    node_cache: _NodeCheckpointBuildCache,
    context: _CheckpointBuildContext,
) -> (
    tuple[
        AlgorithmNode[Any] | None,
        int | None,
        BranchKey | None,
        CheckpointAtomPayload | None,
    ]
    | None
):
    """Return the stored delta parent/link when it is still a live incoming edge."""
    payload = checkpoint_payload_for_reuse_or_none(node.state_handle)
    if not isinstance(payload, DeltaCheckpointStatePayload):
        return None

    scan_started_at = perf_counter()
    try:
        for parent_branch_entry in node_cache.parent_branch_entries:
            if parent_branch_entry.parent_node.id != payload.state_parent_node_id:
                continue
            branch = _stored_or_first_branch_in_stable_order(
                parent_branch_entry,
                preferred_branch_payload=payload.state_parent_branch,
                context=context,
            )
            return (
                parent_branch_entry.parent_node,
                parent_branch_entry.parent_node.id,
                branch,
                _serialize_checkpoint_atom_for_build(branch, context=context),
            )
        return None
    finally:
        context.metrics.reusable_parent_scan_calls += 1
        context.metrics.reusable_parent_scan_s += perf_counter() - scan_started_at


def _build_checkpoint_state_payload(
    *,
    node: AlgorithmNode[Any],
    representative_parent: AlgorithmNode[Any] | None,
    state_codec: IncrementalStateCheckpointCodec[Any],
    metrics: _CheckpointBuildMetrics,
    context: _CheckpointBuildContext,
    node_cache: _NodeCheckpointBuildCache,
) -> AnchorCheckpointStatePayload | DeltaCheckpointStatePayload:
    """Build one explicit anchor-or-delta checkpoint payload for ``node``."""
    state_payload_started_at = perf_counter()
    try:
        reuse_started_at = perf_counter()
        reused_payload = _try_reuse_checkpoint_state_payload(
            node=node,
            representative_parent=representative_parent,
            metrics=metrics,
        )
        metrics.state_payload_reuse_s += perf_counter() - reuse_started_at
        if reused_payload is not None:
            metrics.state_payloads_reused += 1
            return reused_payload
        state_summary_started_at = perf_counter()
        state_summary = _dump_optional_state_summary(
            node.state, state_codec=state_codec
        )
        metrics.state_summary_total_s += perf_counter() - state_summary_started_at
        if _is_anchor_node(node=node, parent_node=representative_parent):
            anchor_started_at = perf_counter()
            anchor_ref = state_codec.dump_anchor_ref(node.state)
            metrics.anchor_state_total_s += perf_counter() - anchor_started_at
            metrics.anchor_payloads_emitted += 1
            return AnchorCheckpointStatePayload(
                anchor_ref=anchor_ref,
                state_summary=state_summary,
            )
        delta_payload = _try_build_delta_state_payload(
            node=node,
            state_codec=state_codec,
            state_summary=state_summary,
            metrics=metrics,
            context=context,
            node_cache=node_cache,
        )
        if delta_payload is not None:
            return delta_payload
        metrics.anchor_fallbacks += 1
        anchor_started_at = perf_counter()
        anchor_ref = state_codec.dump_anchor_ref(node.state)
        metrics.anchor_state_total_s += perf_counter() - anchor_started_at
        metrics.anchor_payloads_emitted += 1
        return AnchorCheckpointStatePayload(
            anchor_ref=anchor_ref,
            state_summary=state_summary,
        )
    finally:
        metrics.state_payload_total_s += perf_counter() - state_payload_started_at


def _try_reuse_checkpoint_state_payload(
    *,
    node: AlgorithmNode[Any],
    representative_parent: AlgorithmNode[Any] | None,
    metrics: _CheckpointBuildMetrics,
) -> AnchorCheckpointStatePayload | DeltaCheckpointStatePayload | None:
    """Return a reusable checkpoint state payload without materializing state."""
    raw_handle = node.state_handle
    payload = checkpoint_payload_for_reuse_or_none(raw_handle)
    if payload is None:
        return None

    if isinstance(payload, AnchorCheckpointStatePayload):
        metrics.anchor_payloads_reused += 1
        metrics.anchor_payloads_emitted += 1
        return payload

    if _delta_reuse_matches_parent(
        representative_parent=representative_parent,
        payload=payload,
    ):
        metrics.delta_payloads_reused += 1
        metrics.delta_payloads_emitted += 1
        return payload

    metrics.state_payload_reuse_rejected += 1
    return None


def _delta_reuse_matches_parent(
    *,
    representative_parent: AlgorithmNode[Any] | None,
    payload: DeltaCheckpointStatePayload,
) -> bool:
    """Return whether a stored delta still matches the chosen representative parent."""
    if representative_parent is None:
        return False
    parent_handle_payload = checkpoint_payload_for_reuse_or_none(
        representative_parent.state_handle
    )
    if parent_handle_payload is None:
        return False
    return payload.state_parent_node_id == representative_parent.id


def _try_build_delta_state_payload(
    *,
    node: AlgorithmNode[Any],
    state_codec: IncrementalStateCheckpointCodec[Any],
    state_summary: CheckpointStateSummary | None,
    metrics: _CheckpointBuildMetrics,
    context: _CheckpointBuildContext,
    node_cache: _NodeCheckpointBuildCache,
) -> DeltaCheckpointStatePayload | None:
    """Return one codec-valid delta payload or ``None`` when anchoring is safer."""
    for candidate_parent, branch in _iter_candidate_state_parent_links(
        node_cache=node_cache,
    ):
        metrics.delta_candidates_attempted += 1
        delta_started_at = perf_counter()
        try:
            delta_ref = state_codec.dump_delta_from_parent(
                parent_state=candidate_parent.state,
                child_state=node.state,
                branch_from_parent=branch,
            )
            metrics.delta_state_total_s += perf_counter() - delta_started_at
        except Exception as exc:  # pylint: disable=broad-exception-caught
            metrics.delta_state_total_s += perf_counter() - delta_started_at
            metrics.delta_candidates_rejected += 1
            checkpoint_logger.debug(
                "delta candidate rejected child=%s parent=%s branch=%r err=%s",
                node.id,
                candidate_parent.id,
                _serialize_checkpoint_atom_for_build(branch, context=context),
                type(exc).__name__,
            )
            continue
        metrics.delta_payloads_emitted += 1
        return DeltaCheckpointStatePayload(
            state_parent_node_id=candidate_parent.id,
            state_parent_branch=_serialize_checkpoint_atom_for_build(
                branch,
                context=context,
            ),
            delta_ref=delta_ref,
            state_summary=state_summary,
        )
    return None


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


def _iter_candidate_state_parent_links(
    *,
    node_cache: _NodeCheckpointBuildCache,
) -> Iterable[tuple[AlgorithmNode[Any], BranchKey]]:
    """Yield candidate state-parent edges in deterministic order."""
    for parent_branch_entry in node_cache.parent_branch_entries:
        for branch in parent_branch_entry.ordered_branches:
            yield parent_branch_entry.parent_node, branch


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
    child_links: Iterable[tuple[Any, AlgorithmNode[Any] | None]],
    *,
    context: _CheckpointBuildContext,
) -> list[LinkedChildCheckpointPayload]:
    """Serialize linked children in stable checkpoint-atom order."""
    metrics = context.metrics
    metrics.linked_children_calls += 1
    linked_children = [
        LinkedChildCheckpointPayload(
            branch_key=_serialize_checkpoint_atom_for_build(
                branch,
                context=context,
            ),
            child_node_id=child.id,
        )
        for branch, child in child_links
        if child is not None
    ]
    sort_started_at = perf_counter()
    linked_children.sort(key=lambda item: (repr(item.branch_key), item.child_node_id))
    metrics.linked_children_sort_calls += 1
    metrics.linked_children_sort_s += perf_counter() - sort_started_at
    return linked_children


def _first_branch_in_stable_order(
    parent_branch_entry: _ParentBranchSerialization,
    *,
    context: _CheckpointBuildContext,
) -> Any:
    """Return the first branch under checkpoint-atom representation order."""
    context.metrics.first_branch_calls += 1
    started_at = perf_counter()
    try:
        return parent_branch_entry.ordered_branches[0]
    finally:
        context.metrics.first_branch_total_s += perf_counter() - started_at


def _stored_or_first_branch_in_stable_order(
    parent_branch_entry: _ParentBranchSerialization,
    *,
    preferred_branch_payload: CheckpointAtomPayload | None,
    context: _CheckpointBuildContext,
) -> Any:
    """Return the stored branch when present, otherwise the stable first branch."""
    context.metrics.stored_or_first_branch_calls += 1
    started_at = perf_counter()
    try:
        for branch, serialized_branch in zip(
            parent_branch_entry.ordered_branches,
            parent_branch_entry.serialized_branches,
            strict=False,
        ):
            if serialized_branch == preferred_branch_payload:
                return branch
        return parent_branch_entry.ordered_branches[0]
    finally:
        context.metrics.stored_or_first_branch_total_s += perf_counter() - started_at


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


def _serialize_optional_value(
    value: Value | None,
    *,
    context: _CheckpointBuildContext,
    value_kind: str,
) -> SerializedValuePayload | None:
    """Serialize one optional Value payload."""
    if value is None:
        return None

    started_at = perf_counter()
    try:
        return _serialize_value_for_build(value, context=context)
    finally:
        elapsed_s = perf_counter() - started_at
        if value_kind == "direct":
            context.metrics.direct_value_serialize_calls += 1
            context.metrics.direct_value_serialize_s += elapsed_s
        else:
            context.metrics.backed_up_value_serialize_calls += 1
            context.metrics.backed_up_value_serialize_s += elapsed_s


def _build_decision_ordering_payload(
    node_eval: Any,
    *,
    context: _CheckpointBuildContext,
) -> DecisionOrderingCheckpointPayload | None:
    """Serialize cached decision-ordering keys when the evaluation exposes them."""
    decision_ordering = getattr(node_eval, "decision_ordering", None)
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
    pv_state = getattr(node_eval, "pv_state", None)
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
    branch_frontier = getattr(node_eval, "branch_frontier", None)
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
    backup_runtime = getattr(node_eval, "backup_runtime", None)
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


def _serialize_branch_collection(
    branches: Iterable[Any],
    *,
    context: _CheckpointBuildContext,
    atom_scope: str = "default",
) -> list[CheckpointAtomPayload]:
    """Serialize a branch collection in deterministic checkpoint-atom order."""
    metrics = context.metrics
    metrics.branch_collection_calls += 1
    started_at = perf_counter()
    try:
        atom_serializer = _serialize_checkpoint_atom_for_build
        if atom_scope == "evaluation":
            atom_serializer = _serialize_evaluation_atom_for_build
        serialized_branches = [
            atom_serializer(branch, context=context) for branch in branches
        ]
        sort_started_at = perf_counter()
        serialized_branches.sort(key=repr)
        metrics.branch_collection_sort_calls += 1
        metrics.branch_collection_sort_s += perf_counter() - sort_started_at
        return serialized_branches
    finally:
        metrics.branch_collection_total_s += perf_counter() - started_at


def _serialize_optional_atom(
    value: object | None,
    *,
    context: _CheckpointBuildContext | None = None,
) -> CheckpointAtomPayload | None:
    """Serialize a small optional atom payload."""
    if value is None:
        return None
    if context is None:
        return serialize_checkpoint_atom(value)
    return _serialize_checkpoint_atom_for_build(value, context=context)


def _serialize_optional_evaluation_atom(
    value: object | None,
    *,
    context: _CheckpointBuildContext,
) -> CheckpointAtomPayload | None:
    """Serialize one optional atom while attributing metrics to evaluation work."""
    if value is None:
        return None
    return _serialize_evaluation_atom_for_build(value, context=context)


def _serialize_parent_branches(
    parent_node: AlgorithmNode[Any],
    branches: Iterable[BranchKey],
    *,
    context: _CheckpointBuildContext,
) -> _ParentBranchSerialization:
    """Return stable ordered branch serializations for one parent edge set."""
    serialized_pairs = [
        (
            branch,
            _serialize_checkpoint_atom_for_build(branch, context=context),
        )
        for branch in branches
    ]
    sort_started_at = perf_counter()
    serialized_pairs.sort(key=lambda item: repr(item[1]))
    context.metrics.branch_collection_sort_calls += 1
    context.metrics.branch_collection_sort_s += perf_counter() - sort_started_at
    return _ParentBranchSerialization(
        parent_node=parent_node,
        ordered_branches=tuple(branch for branch, _payload in serialized_pairs),
        serialized_branches=tuple(payload for _branch, payload in serialized_pairs),
    )


def _serialize_checkpoint_atom_for_build(
    value: object,
    *,
    context: _CheckpointBuildContext,
) -> CheckpointAtomPayload:
    """Serialize checkpoint atoms with build-local memoization for hashable atoms."""
    metrics = context.metrics
    metrics.atom_serialize_calls += 1
    try:
        hash(value)
    except TypeError:
        started_at = perf_counter()
        payload = serialize_checkpoint_atom(value)
        metrics.atom_serialize_total_s += perf_counter() - started_at
        return payload

    if value in context.atom_serialization_cache:
        metrics.atom_serialize_cache_hits += 1
        return context.atom_serialization_cache[value]

    metrics.atom_serialize_cache_misses += 1
    started_at = perf_counter()
    payload = serialize_checkpoint_atom(value)
    metrics.atom_serialize_total_s += perf_counter() - started_at
    context.atom_serialization_cache[value] = payload
    return payload


def _serialize_evaluation_atom_for_build(
    value: object,
    *,
    context: _CheckpointBuildContext,
) -> CheckpointAtomPayload:
    """Serialize one atom while attributing metrics to evaluation payload work."""
    metrics = context.metrics
    metrics.evaluation_atom_serialize_calls += 1
    atom_cache_hits_before = metrics.atom_serialize_cache_hits
    atom_cache_misses_before = metrics.atom_serialize_cache_misses
    atom_total_before = metrics.atom_serialize_total_s
    payload = _serialize_checkpoint_atom_for_build(value, context=context)
    metrics.evaluation_atom_serialize_total_s += (
        metrics.atom_serialize_total_s - atom_total_before
    )
    metrics.evaluation_atom_serialize_cache_hits += (
        metrics.atom_serialize_cache_hits - atom_cache_hits_before
    )
    metrics.evaluation_atom_serialize_cache_misses += (
        metrics.atom_serialize_cache_misses - atom_cache_misses_before
    )
    return payload


def _serialize_value_for_build(
    value: Value,
    *,
    context: _CheckpointBuildContext,
) -> SerializedValuePayload:
    """Serialize one Value with build-local caching and detailed timing."""
    metrics = context.metrics
    metrics.serialize_value_calls += 1

    identity_key = id(value)
    cached_identity_entry = context.value_identity_serialization_cache.get(identity_key)
    if cached_identity_entry is not None and cached_identity_entry[0] is value:
        metrics.serialize_value_cache_hits += 1
        return cached_identity_entry[1]

    metrics.serialize_value_cache_misses += 1
    payload = _serialize_value_uncached_for_build(value, context=context)
    context.value_identity_serialization_cache[identity_key] = (value, payload)
    return payload


def _serialize_value_uncached_for_build(
    value: Value,
    *,
    context: _CheckpointBuildContext,
) -> SerializedValuePayload:
    """Serialize one Value payload while recording validation and atom work."""
    metrics = context.metrics
    started_at = perf_counter()
    try:
        validation_started_at = perf_counter()
        validated_value = canonical_value.validate_value_semantics(value)
        metrics.value_semantic_validation_calls += 1
        metrics.value_semantic_validation_s += perf_counter() - validation_started_at
        return SerializedValuePayload(
            score=validated_value.score,
            certainty=validated_value.certainty.name,
            over_event=_serialize_over_event_for_build(
                validated_value.over_event,
                context=context,
            ),
            line=(
                [
                    _serialize_evaluation_atom_for_build(branch, context=context)
                    for branch in validated_value.line
                ]
                if validated_value.line is not None
                else None
            ),
        )
    finally:
        metrics.serialize_value_total_s += perf_counter() - started_at


def _serialize_over_event_for_build(
    over_event: object | None,
    *,
    context: _CheckpointBuildContext,
) -> SerializedOverEventPayload | None:
    """Serialize over-event metadata while attributing atom work to evaluation."""
    return serialize_over_event_with(
        over_event,
        atom_serializer=lambda atom: _serialize_optional_evaluation_atom(
            atom,
            context=context,
        ),
    )


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


def _maybe_dump_rng_state(search: TreeExploration[Any]) -> object | None:
    """Return directly exposed selector RNG state when available."""
    random_generator = getattr(search.node_selector, "random_generator", None)
    if isinstance(random_generator, Random):
        return random_generator.getstate()

    # TreeExploration does not retain the explore() RNG. Selector-specific RNG
    # restoration can be made more complete with selector checkpoint payloads.
    return None


def _build_selector_state_payload(
    search: TreeExploration[Any],
) -> SelectorCheckpointPayload | None:
    """Serialize optional selector-private checkpoint state when supported."""
    objective = search.tree.root_node.tree_evaluation.required_objective
    if not _is_single_agent_objective(objective):
        return None
    if not isinstance(search.node_selector, StatefulNodeSelector):
        return None
    stateful_selector = cast(
        "StatefulNodeSelector[AlgorithmNode[Any]]",
        search.node_selector,
    )
    stateful_selector.refresh_state_for_checkpoint(
        tree=search.tree,
        objective=objective,
        latest_tree_expansions=search.latest_tree_expansions,
    )
    return stateful_selector.build_checkpoint_payload(objective)


def _is_single_agent_objective(
    objective: object,
) -> TypeGuard[SingleAgentMaxObjective[Any]]:
    """Return whether one objective is a typed single-agent objective."""
    return isinstance(objective, SingleAgentMaxObjective)


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


__all__ = ["build_search_checkpoint_payload"]
