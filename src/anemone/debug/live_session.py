"""Incremental live session recording helpers for browser debug viewing.

A live session is the on-disk state written while exploration is running. It is
distinct from an offline replay bundle: the session JSON stays mutable until
``finalize()`` marks the run complete.
"""

# pylint: disable=duplicate-code
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .export import (
    export_snapshot_entry,
    export_snapshot_json,
    trace_snapshot_filename,
    trace_snapshot_metadata_filename,
)
from .html_templates import render_replay_index_html
from .recording import (
    DebugTimelineEntry,
    DebugTrace,
    SnapshotPolicy,
    SnapshotProvider,
)
from .replay_bundle import build_replay_entry_payload, write_replay_payload
from .sink import SearchDebugSink

if TYPE_CHECKING:
    from .events import SearchDebugEvent
    from .live_control import DebugSessionController

_DEFAULT_SNAPSHOT_DIRECTORY = "snapshots"


class MissingLiveSessionSnapshotError(ValueError):
    """Tried to export a live-session entry that has no attached snapshot."""

    def __init__(self) -> None:
        """Initialize the error with the default missing-snapshot message."""
        super().__init__("Cannot export a snapshot for an entry without a snapshot.")


class LiveDebugSessionRecorder(SearchDebugSink):
    """Record a live debug session into an incrementally updated bundle."""

    def __init__(
        self,
        directory: str | Path,
        *,
        snapshot_provider: SnapshotProvider | None = None,
        snapshot_policy: SnapshotPolicy | None = None,
        snapshot_format: str = "svg",
        html_template: str | None = None,
        controller: DebugSessionController | None = None,
    ) -> None:
        """Initialize a live session bundle rooted at ``directory``."""
        self._directory = Path(directory)
        self._snapshot_directory = self._directory / _DEFAULT_SNAPSHOT_DIRECTORY
        self._snapshot_provider = snapshot_provider
        self._snapshot_policy = snapshot_policy
        self._snapshot_format = snapshot_format
        self._html_template = html_template
        self._controller = controller
        self._entries: list[DebugTimelineEntry] = []
        self._is_complete = False

        self._directory.mkdir(parents=True, exist_ok=True)
        self._snapshot_directory.mkdir(parents=True, exist_ok=True)
        self.write_index_html()
        self.write_session_json()

    def emit(self, event: SearchDebugEvent) -> None:
        """Record one event and rewrite the live session payload."""
        snapshot = None
        if self._snapshot_provider is not None and self._should_capture_snapshot(event):
            snapshot = self._snapshot_provider(event)

        breakpoint_hit = None
        if self._controller is not None:
            matching_breakpoint = self._controller.handle_event(event)
            if matching_breakpoint is not None:
                breakpoint_hit = matching_breakpoint.id

        entry = DebugTimelineEntry(
            index=len(self._entries),
            event=event,
            snapshot=snapshot,
            breakpoint_hit=breakpoint_hit,
        )
        self._entries.append(entry)
        self._is_complete = False

        if snapshot is not None:
            self._export_snapshot(entry)

        self.write_session_json()

    def finalize(self) -> None:
        """Mark the live session as complete and rewrite the session payload.

        Browser viewers can follow the session while it is still incomplete.
        Finalization only flips the explicit completion marker written to
        ``session.json``.
        """
        self._is_complete = True
        self.write_session_json()

    def to_trace(self) -> DebugTrace:
        """Return an immutable trace view of the currently recorded session."""
        return DebugTrace(entries=tuple(self._entries))

    def write_index_html(self) -> Path:
        """Ensure the viewer HTML exists in the session directory."""
        html_path = self._directory / "index.html"
        if html_path.exists() and self._html_template is None:
            return html_path

        html_path.write_text(
            self._html_template or render_replay_index_html(),
            encoding="utf-8",
        )
        return html_path

    def write_session_json(self) -> Path:
        """Rewrite the session payload JSON for the current live state."""
        return write_replay_payload(self._payload(), self._directory / "session.json")

    def _payload(self) -> dict[str, Any]:
        """Build the JSON-friendly payload describing the current live session."""
        return {
            "entry_count": len(self._entries),
            "entries": [
                build_replay_entry_payload(
                    entry,
                    snapshot_format=self._snapshot_format,
                    snapshot_directory=_DEFAULT_SNAPSHOT_DIRECTORY,
                )
                for entry in self._entries
            ],
            "is_live": True,
            "is_complete": self._is_complete,
        }

    def _should_capture_snapshot(self, event: SearchDebugEvent) -> bool:
        """Return whether ``event`` should include a snapshot."""
        if self._snapshot_provider is None:
            return False
        if self._snapshot_policy is None:
            return True
        return self._snapshot_policy(event)

    def _export_snapshot(self, entry: DebugTimelineEntry) -> Path:
        """Export the snapshot attached to ``entry`` if not already written."""
        if entry.snapshot is None:
            raise MissingLiveSessionSnapshotError
        snapshot_path = self._snapshot_directory / trace_snapshot_filename(
            entry,
            format_str=self._snapshot_format,
        )
        if snapshot_path.exists():
            metadata_path = self._snapshot_directory / trace_snapshot_metadata_filename(
                entry
            )
            if not metadata_path.exists():
                export_snapshot_json(entry.snapshot, metadata_path)
            return snapshot_path

        exported_snapshot_path = export_snapshot_entry(
            entry.snapshot,
            snapshot_path,
            format_str=self._snapshot_format,
        )
        export_snapshot_json(
            entry.snapshot,
            self._snapshot_directory / trace_snapshot_metadata_filename(entry),
        )
        return exported_snapshot_path


__all__ = ["LiveDebugSessionRecorder", "MissingLiveSessionSnapshotError"]
