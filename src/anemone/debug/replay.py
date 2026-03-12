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
    if isinstance(event, SearchIterationStarted):
        return f"iteration {event.iteration_index} started"
    if isinstance(event, SearchIterationCompleted):
        return f"iteration {event.iteration_index} completed"
    if isinstance(event, NodeSelected):
        return f"node {event.node_id} selected"
    if isinstance(event, NodeOpeningPlanned):
        return f"node {event.node_id} opening planned ({event.branch_count} branches)"
    if isinstance(event, ChildLinked):
        already = " (already present)" if event.was_already_present else ""
        return f"linked {event.parent_id} --{event.branch_key}--> {event.child_id}{already}"
    if isinstance(event, DirectValueAssigned):
        return f"node {event.node_id} direct value = {event.value_repr}"
    if isinstance(event, BackupStarted):
        return f"node {event.node_id} backup started"
    if isinstance(event, BackupFinished):
        return (
            f"node {event.node_id} backup finished "
            "("
            f"value_changed={event.value_changed}, "
            f"pv_changed={event.pv_changed}, "
            f"over_changed={event.over_changed}"
            ")"
        )

    return event.__class__.__name__


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


__all__ = ["TraceReplayView", "format_debug_event"]
