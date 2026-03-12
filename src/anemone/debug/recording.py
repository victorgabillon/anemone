"""Replayable debug timeline recording utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .events import SearchDebugEvent
from .model import DebugTreeSnapshot
from .sink import SearchDebugSink
from .snapshot_adapter import TreeSnapshotAdapter

type SnapshotProvider = Callable[[SearchDebugEvent], DebugTreeSnapshot | None]
type SnapshotPolicy = Callable[[SearchDebugEvent], bool]


@dataclass(frozen=True)
class DebugTimelineEntry:
    """One recorded step in a debug trace."""

    index: int
    event: SearchDebugEvent
    snapshot: DebugTreeSnapshot | None = None
    breakpoint_hit: str | None = None


@dataclass(frozen=True)
class DebugTrace:
    """Replayable ordered debug trace for one search run."""

    entries: tuple[DebugTimelineEntry, ...]

    def __iter__(self) -> Iterator[DebugTimelineEntry]:
        """Iterate over recorded timeline entries."""
        return iter(self.entries)

    def __len__(self) -> int:
        """Return the number of recorded entries."""
        return len(self.entries)

    def entry_at(self, index: int) -> DebugTimelineEntry:
        """Return the recorded entry at ``index``."""
        return self.entries[index]

    def events_of_type(self, event_type: type[Any]) -> tuple[DebugTimelineEntry, ...]:
        """Return entries whose event matches ``event_type``."""
        return tuple(
            entry for entry in self.entries if isinstance(entry.event, event_type)
        )

    def last_snapshot(self) -> DebugTreeSnapshot | None:
        """Return the last recorded snapshot, if any."""
        for entry in reversed(self.entries):
            if entry.snapshot is not None:
                return entry.snapshot
        return None


class DebugTimelineRecorder(SearchDebugSink):
    """Record debug events into an immutable replayable trace."""

    def __init__(
        self,
        *,
        snapshot_provider: SnapshotProvider | None = None,
        snapshot_policy: SnapshotPolicy | None = None,
    ) -> None:
        """Initialize the recorder with optional snapshot capture callbacks."""
        self._entries: list[DebugTimelineEntry] = []
        self._snapshot_provider = snapshot_provider
        self._snapshot_policy = snapshot_policy

    def emit(self, event: SearchDebugEvent) -> None:
        """Record ``event`` and capture an optional snapshot."""
        snapshot = None
        if self._snapshot_provider is not None and self._should_capture_snapshot(event):
            snapshot = self._snapshot_provider(event)

        self._entries.append(
            DebugTimelineEntry(
                index=len(self._entries),
                event=event,
                snapshot=snapshot,
            )
        )

    def to_trace(self) -> DebugTrace:
        """Freeze the currently recorded entries into a ``DebugTrace``."""
        return DebugTrace(entries=tuple(self._entries))

    def _should_capture_snapshot(self, event: SearchDebugEvent) -> bool:
        """Return whether ``event`` should include a tree snapshot."""
        if self._snapshot_provider is None:
            return False
        if self._snapshot_policy is None:
            return True
        return self._snapshot_policy(event)


def make_tree_snapshot_provider(
    *,
    root_getter: Callable[[], Any],
    adapter: TreeSnapshotAdapter | None = None,
) -> SnapshotProvider:
    """Return a snapshot provider that captures the current tree root lazily."""
    snapshot_adapter = adapter or TreeSnapshotAdapter()

    def snapshot_provider(event: SearchDebugEvent) -> DebugTreeSnapshot | None:
        del event
        root = root_getter()
        if root is None:
            return None
        return snapshot_adapter.snapshot(root)

    return snapshot_provider


__all__ = [
    "DebugTimelineEntry",
    "DebugTimelineRecorder",
    "DebugTrace",
    "SnapshotPolicy",
    "SnapshotProvider",
    "make_tree_snapshot_provider",
]
