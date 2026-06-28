"""Build checkpoint payloads for anchor and delta node state."""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone.utils.logger import checkpoint_logger

from ._protocols import (
    CheckpointParentBranchPayloadCodec,
    CheckpointStateSummary,
    IncrementalStateCheckpointCodec,
    StateCheckpointSummaryCodec,
)
from .build_atoms import _serialize_checkpoint_atom_for_build
from .build_context import (
    CHECKPOINT_ANCHOR_DEPTH_STRIDE,
    _CheckpointBuildContext,
    _CheckpointBuildMetrics,
    _NodeCheckpointBuildCache,
    _ParentBranchSerialization,
)
from .payloads import (
    AnchorCheckpointStatePayload,
    CheckpointAtomPayload,
    DeltaCheckpointStatePayload,
)
from .state_handles import checkpoint_payload_for_reuse_or_none

if TYPE_CHECKING:
    from collections.abc import Iterable

    from valanga import BranchKey

    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode


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
            state_parent_branch=_dump_state_parent_branch_for_checkpoint(
                branch_from_parent=branch,
                state_codec=state_codec,
                context=context,
            ),
            delta_ref=delta_ref,
            state_summary=state_summary,
        )
    return None


def _dump_state_parent_branch_for_checkpoint(
    *,
    branch_from_parent: object,
    state_codec: IncrementalStateCheckpointCodec[Any],
    context: _CheckpointBuildContext,
) -> CheckpointAtomPayload | None:
    """Return the payload stored as ``state_parent_branch`` for one delta node."""
    if isinstance(state_codec, CheckpointParentBranchPayloadCodec):
        return cast(
            "CheckpointAtomPayload | None",
            cast(
                "CheckpointParentBranchPayloadCodec",
                state_codec,
            ).dump_state_parent_branch_for_checkpoint(cast("Any", branch_from_parent)),
        )
    return _serialize_checkpoint_atom_for_build(branch_from_parent, context=context)


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
