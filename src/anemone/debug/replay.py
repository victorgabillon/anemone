"""Replay helpers for browsing recorded debug traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .events import (
    BackupFinished,
    BackupStarted,
    ChildLinked,
    DirectValueAssigned,
    NodeOpeningPlanned,
    NodeSelected,
    SearchDebugEvent,
    SearchIterationCompleted,
    SearchIterationStarted,
)

if TYPE_CHECKING:
    from .model import DebugTreeSnapshot
    from .recording import DebugTimelineEntry, DebugTrace


def format_debug_event(event: SearchDebugEvent) -> str:
    """Return a compact developer-facing summary line for one debug event."""
    return _format_debug_event_object(event)


def build_event_fields_payload(event: SearchDebugEvent) -> dict[str, object]:
    """Return a structured JSON-friendly field map for one debug event."""
    return _build_event_fields_object(event)


def _format_debug_event_object(event: object) -> str:
    """Return a readable debug summary for known events or a class-name fallback."""
    match event:
        case SearchIterationStarted(iteration_index=iteration_index):
            return f"iteration {iteration_index} started"
        case SearchIterationCompleted(iteration_index=iteration_index):
            return f"iteration {iteration_index} completed"
        case NodeSelected(node_id=node_id):
            return f"node {node_id} selected"
        case NodeOpeningPlanned(node_id=node_id, branch_count=branch_count):
            return f"node {node_id} opening planned ({branch_count} branches)"
        case ChildLinked(
            parent_id=parent_id,
            child_id=child_id,
            branch_key=branch_key,
            was_already_present=was_already_present,
        ):
            already = " (already present)" if was_already_present else ""
            return f"linked {parent_id} --{branch_key}--> {child_id}{already}"
        case DirectValueAssigned(node_id=node_id, value_repr=value_repr):
            return f"node {node_id} direct value = {value_repr}"
        case BackupStarted(node_id=node_id):
            return f"node {node_id} backup started"
        case BackupFinished(
            node_id=node_id,
            value_changed=value_changed,
            pv_changed=pv_changed,
            over_changed=over_changed,
        ):
            return (
                f"node {node_id} backup finished "
                "("
                f"value_changed={value_changed}, "
                f"pv_changed={pv_changed}, "
                f"over_changed={over_changed}"
                ")"
            )
        case _:
            return event.__class__.__name__


def _build_event_fields_object(event: object) -> dict[str, object]:
    """Return structured event fields for browser-side navigation helpers."""
    match event:
        case SearchIterationStarted(iteration_index=iteration_index):
            return {"iteration_index": iteration_index}
        case SearchIterationCompleted(iteration_index=iteration_index):
            return {"iteration_index": iteration_index}
        case NodeSelected(node_id=node_id):
            return {"node_id": node_id}
        case NodeOpeningPlanned(node_id=node_id, branch_count=branch_count):
            return {"node_id": node_id, "branch_count": branch_count}
        case ChildLinked(
            parent_id=parent_id,
            child_id=child_id,
            branch_key=branch_key,
            was_already_present=was_already_present,
        ):
            return {
                "parent_id": parent_id,
                "child_id": child_id,
                "branch_key": branch_key,
                "was_already_present": was_already_present,
            }
        case DirectValueAssigned(node_id=node_id, value_repr=value_repr):
            return {"node_id": node_id, "value_repr": value_repr}
        case BackupStarted(node_id=node_id):
            return {"node_id": node_id}
        case BackupFinished(
            node_id=node_id,
            value_changed=value_changed,
            pv_changed=pv_changed,
            over_changed=over_changed,
        ):
            return {
                "node_id": node_id,
                "value_changed": value_changed,
                "pv_changed": pv_changed,
                "over_changed": over_changed,
            }
        case _:
            return {}


@dataclass(frozen=True)
class TraceReplayView:
    """Navigation helper for replaying a ``DebugTrace`` timeline."""

    trace: DebugTrace

    def __len__(self) -> int:
        """Return the total number of timeline entries."""
        return len(self.trace)

    def entry_at(self, index: int) -> DebugTimelineEntry:
        """Return the timeline entry at ``index``."""
        return self.trace.entry_at(index)

    def snapshot_at(self, index: int) -> DebugTreeSnapshot | None:
        """Return the snapshot attached to ``index``, if present."""
        return self.entry_at(index).snapshot

    def nearest_snapshot_at_or_before(self, index: int) -> DebugTreeSnapshot | None:
        """Return the closest available snapshot at or before ``index``."""
        for entry_index in range(index, -1, -1):
            snapshot = self.snapshot_at(entry_index)
            if snapshot is not None:
                return snapshot
        return None

    def entries_with_snapshots(self) -> tuple[DebugTimelineEntry, ...]:
        """Return only timeline entries that include a snapshot payload."""
        return tuple(entry for entry in self.trace if entry.snapshot is not None)

    def event_summary_at(self, index: int) -> str:
        """Return one compact summary line for the entry at ``index``."""
        return format_debug_event(self.entry_at(index).event)


__all__ = ["TraceReplayView", "build_event_fields_payload", "format_debug_event"]
