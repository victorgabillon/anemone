"""Checkpoint conversion and validation helpers for Linoo."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.nodes.itree_node import ITreeNode

from .errors import invalid_linoo_checkpoint_payload_error
from .runtime_state import LinooDepthStats, LinooNodeState
from .types import LINOO_DEFAULT_NODE_STATUS, LINOO_NODE_STATUSES, LinooNodeStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from valanga.evaluations import Value

    from anemone import trees
    from anemone.checkpoints.payloads import (
        LinooCandidatesByDepthCheckpointPayload,
        LinooDepthStatsCheckpointPayload,
        LinooNodeStateCheckpointPayload,
        LinooSelectorCheckpointPayload,
    )
    from anemone.node_selector.linoo.candidate_heap import (
        LinooCandidateHeap,
        LinooHeapEntry,
    )


def depth_stats_payload_from_cache(
    depth_stats_by_depth: Mapping[int, LinooDepthStats],
) -> list[LinooDepthStatsCheckpointPayload]:
    """Serialize cached depth stats in stable depth order."""
    from anemone.checkpoints.payloads import (  # pylint: disable=import-outside-toplevel
        LinooDepthStatsCheckpointPayload,
    )

    return [
        LinooDepthStatsCheckpointPayload(
            depth=depth,
            total_nodes=stats.total_nodes,
            opened_count=stats.opened_count,
            frontier_count=stats.frontier_count,
            terminal_count=stats.terminal_count,
            exact_count=stats.exact_count,
            uncached_terminal_candidates=stats.uncached_terminal_candidates,
            non_openable_count=stats.non_openable_count,
        )
        for depth, stats in sorted(depth_stats_by_depth.items())
    ]


def node_states_payload_from_cache(
    node_state_by_id: Mapping[int, LinooNodeState],
) -> list[LinooNodeStateCheckpointPayload]:
    """Serialize cached node classifications without live node objects."""
    from anemone.checkpoints.payloads import (  # pylint: disable=import-outside-toplevel
        LinooNodeStateCheckpointPayload,
    )

    return [
        LinooNodeStateCheckpointPayload(
            node_id=node_id,
            depth=node_state.depth,
            status=node_state.status,
        )
        for node_id, node_state in sorted(node_state_by_id.items())
    ]


def candidate_payloads_from_cache(
    *,
    candidates_by_depth: Mapping[int, list[LinooHeapEntry]],
    node_state_by_id: Mapping[int, LinooNodeState],
    candidate_versions_by_node_id: Mapping[int, int],
) -> list[LinooCandidatesByDepthCheckpointPayload]:
    """Serialize non-stale candidate heap entries without mutating heaps."""
    from anemone.checkpoints.payloads import (  # pylint: disable=import-outside-toplevel
        LinooCandidateCheckpointPayload,
        LinooCandidatesByDepthCheckpointPayload,
    )

    payloads: list[LinooCandidatesByDepthCheckpointPayload] = []
    for depth, heap in sorted(candidates_by_depth.items()):
        candidates = [
            LinooCandidateCheckpointPayload(
                node_id=node_id,
                depth=depth,
                priority=-priority_key,
                version=version,
            )
            for priority_key, node_id, version in heap
            if should_serialize_candidate(
                depth=depth,
                node_id=node_id,
                version=version,
                node_state_by_id=node_state_by_id,
                candidate_versions_by_node_id=candidate_versions_by_node_id,
            )
        ]
        if candidates:
            payloads.append(
                LinooCandidatesByDepthCheckpointPayload(
                    depth=depth,
                    candidates=sorted(
                        candidates,
                        key=lambda candidate: (
                            -candidate.priority,
                            candidate.node_id,
                            candidate.version,
                        ),
                    ),
                )
            )
    return payloads


def should_serialize_candidate(
    *,
    depth: int,
    node_id: int,
    version: int,
    node_state_by_id: Mapping[int, LinooNodeState],
    candidate_versions_by_node_id: Mapping[int, int],
) -> bool:
    """Return whether one heap entry is current enough to checkpoint."""
    node_state = node_state_by_id.get(node_id)
    return (
        node_state is not None
        and node_state.status == "frontier"
        and node_state.depth == depth
        and version == candidate_versions_by_node_id.get(node_id, 0)
    )


def coerce_node_id(node_or_id: object) -> int:
    """Return one node id from an int or a legacy node-bearing field."""
    if isinstance(node_or_id, int):
        return node_or_id
    node_id = getattr(node_or_id, "id", None)
    if isinstance(node_id, int):
        return node_id
    raise invalid_linoo_checkpoint_payload_error()


def payload_node_id(payload: object) -> int:
    """Return one payload node id, accepting legacy node-bearing schemas."""
    node_id = getattr(payload, "node_id", None)
    if isinstance(node_id, int):
        return node_id
    legacy_node = getattr(payload, "node", None)
    if legacy_node is not None:
        return coerce_node_id(legacy_node)
    raise invalid_linoo_checkpoint_payload_error()


def payload_last_selected_node_id(
    payload: LinooSelectorCheckpointPayload,
) -> int | None:
    """Return the last-selected id, accepting legacy node-bearing payloads."""
    last_selected_node_id = getattr(payload, "last_selected_node_id", None)
    if last_selected_node_id is None:
        legacy_last_selected = getattr(payload, "last_selected_node", None)
        if legacy_last_selected is None:
            return None
        return coerce_node_id(legacy_last_selected)
    return coerce_node_id(last_selected_node_id)


def restore_node_states_from_payload[NodeT: ITreeNode[Any]](
    *,
    tree: trees.Tree[NodeT],
    nodes_by_id: Mapping[int, NodeT],
    payload: LinooSelectorCheckpointPayload,
    classify_node: Callable[[NodeT], LinooNodeStatus],
) -> dict[int, LinooNodeState]:
    """Restore and validate cached node states from payload."""
    node_states: dict[int, LinooNodeState] = {}
    seen_node_ids: set[int] = set()
    for node_payload in payload.node_states:
        node_id = payload_node_id(node_payload)
        if node_id in seen_node_ids:
            raise invalid_linoo_checkpoint_payload_error()
        seen_node_ids.add(node_id)
        if node_payload.status not in LINOO_NODE_STATUSES:
            raise invalid_linoo_checkpoint_payload_error()
        node = nodes_by_id[node_id]
        depth = tree.node_depth(node)
        if node_payload.depth != depth:
            raise invalid_linoo_checkpoint_payload_error()
        status = classify_node(node)
        if node_payload.status != status:
            raise invalid_linoo_checkpoint_payload_error()
        set_restored_node_state_if_non_default(
            node_states=node_states,
            node_id=node_id,
            state=LinooNodeState(
                node_id=node_id,
                depth=depth,
                status=status,
            ),
        )
    for node_id, node in nodes_by_id.items():
        if node_id in seen_node_ids:
            continue
        status = classify_node(node)
        if status != LINOO_DEFAULT_NODE_STATUS:
            raise invalid_linoo_checkpoint_payload_error()
    return node_states


def set_restored_node_state_if_non_default(
    *,
    node_states: dict[int, LinooNodeState],
    node_id: int,
    state: LinooNodeState,
) -> None:
    """Store one restored state only when absence cannot represent it."""
    if state.is_default():
        node_states.pop(node_id, None)
        return
    node_states[node_id] = state


def depth_stats_from_tree_and_node_states[NodeT: ITreeNode[Any]](
    *,
    tree: trees.Tree[NodeT],
    nodes_by_id: Mapping[int, NodeT],
    node_states: Mapping[int, LinooNodeState],
) -> dict[int, LinooDepthStats]:
    """Rebuild depth accounting, treating missing node states as opened."""
    depth_stats_by_depth: dict[int, LinooDepthStats] = {}
    for node_id, node in nodes_by_id.items():
        node_state = node_states.get(node_id)
        if node_state is None:
            node_state = LinooNodeState(
                node_id=node_id,
                depth=tree.node_depth(node),
                status=LINOO_DEFAULT_NODE_STATUS,
            )
        stats = depth_stats_by_depth.setdefault(
            node_state.depth,
            LinooDepthStats(),
        )
        stats.total_nodes += 1
        stats.increment(node_state.status)
    return depth_stats_by_depth


def depth_stats_match_payload(
    depth_stats_by_depth: Mapping[int, LinooDepthStats],
    payloads: list[LinooDepthStatsCheckpointPayload],
) -> bool:
    """Return whether restored depth stats exactly match payload stats."""
    payload_by_depth = {payload.depth: payload for payload in payloads}
    if set(payload_by_depth) != set(depth_stats_by_depth):
        return False
    return all(
        stats.total_nodes == payload_by_depth[depth].total_nodes
        and stats.opened_count == payload_by_depth[depth].opened_count
        and stats.frontier_count == payload_by_depth[depth].frontier_count
        and stats.terminal_count == payload_by_depth[depth].terminal_count
        and stats.exact_count == payload_by_depth[depth].exact_count
        and stats.uncached_terminal_candidates
        == payload_by_depth[depth].uncached_terminal_candidates
        and stats.non_openable_count == payload_by_depth[depth].non_openable_count
        for depth, stats in depth_stats_by_depth.items()
    )


def frontier_ids_from_node_states(
    node_states: Mapping[int, LinooNodeState],
) -> dict[int, set[int]]:
    """Rebuild frontier id buckets from restored node states."""
    frontier_ids_by_depth: dict[int, set[int]] = {}
    for node_id, node_state in node_states.items():
        if node_state.status == "frontier":
            frontier_ids_by_depth.setdefault(node_state.depth, set()).add(node_id)
    return frontier_ids_by_depth


def restore_candidate_payloads[NodeT: ITreeNode[Any]](
    *,
    tree: trees.Tree[NodeT],
    payload: LinooSelectorCheckpointPayload,
    nodes_by_id: Mapping[int, NodeT],
    node_state_by_id: Mapping[int, LinooNodeState],
    candidate_heap: LinooCandidateHeap,
    candidate_value_or_none: Callable[[NodeT], Value | None],
    candidate_signature: Callable[[NodeT, Value], object],
) -> None:
    """Restore valid candidate heap entries and discard stale ones."""
    for depth_payload in payload.candidates_by_depth:
        for candidate in depth_payload.candidates:
            node_id = payload_node_id(candidate)
            if candidate.depth != depth_payload.depth:
                continue
            node = nodes_by_id.get(node_id)
            if node is None:
                continue
            node_state = node_state_by_id.get(node_id)
            if (
                node_state is None
                or node_state.status != "frontier"
                or node_state.depth != candidate.depth
                or tree.node_depth(node) != candidate.depth
            ):
                continue
            current_version = candidate_heap.candidate_versions_by_node_id.get(
                node_id,
                candidate.version,
            )
            if candidate.version != current_version:
                continue
            candidate_value = candidate_value_or_none(node)
            if candidate_value is None:
                continue
            candidate_heap.restore_candidate(
                depth=candidate.depth,
                node_id=node_id,
                priority=candidate.priority,
                version=candidate.version,
                signature=candidate_signature(node, candidate_value),
            )
