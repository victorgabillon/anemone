"""Build runtime checkpoint payloads from live Anemone searches."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import fields, is_dataclass
from random import Random
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone.node_evaluation.common import canonical_value
from anemone.utils.logger import checkpoint_logger
from anemone.utils.small_tools import Interval

from ._protocols import (
    CheckpointParentBranchPayloadCodec,
    CheckpointStateSummary,
    IncrementalStateCheckpointCodec,
    StateCheckpointSummaryCodec,
)
from .build_context import (
    CHECKPOINT_ANCHOR_DEPTH_STRIDE,
    _CheckpointBuildContext,
    _CheckpointBuildMetrics,
    _log_checkpoint_build_metrics,
    _maybe_log_checkpoint_codec_profile,
    _maybe_reset_checkpoint_profile,
    _NodeCheckpointBuildCache,
    _ParentBranchSerialization,
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
    SerializedOverEventPayload,
    SerializedValuePayload,
    TreeCheckpointPayload,
)
from .selector_payloads import _build_selector_state_payload
from .state_handles import checkpoint_payload_for_reuse_or_none
from .tree_expansions_payloads import (
    _build_latest_tree_expansions_payload,
)
from .value_serialization import (
    serialize_checkpoint_atom,
    serialize_over_event_with,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone.node_evaluation.common.branch_frontier import BranchFrontierState
    from anemone.node_evaluation.common.principal_variation import (
        PrincipalVariationState,
    )
    from anemone.node_evaluation.tree.decision_ordering import DecisionOrderingState
    from anemone.node_evaluation.tree.top2_exactness_pv_runtime import (
        Top2ExactnessPvRuntime,
    )
    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
    from anemone.tree_exploration import TreeExploration


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
            for parent_node, branch_keys in node.iter_parent_items()
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


__all__ = ["build_search_checkpoint_payload"]
