"""Replay bundle helpers for exporting browser-viewable debug traces."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from .export import (
    export_snapshot_entry,
    export_snapshot_json,
    trace_snapshot_filename,
    trace_snapshot_metadata_filename,
)
from .html_templates import render_replay_index_html
from .replay import format_debug_event

if TYPE_CHECKING:
    from .recording import DebugTimelineEntry, DebugTrace

_DEFAULT_SNAPSHOT_DIRECTORY = "snapshots"


def build_replay_payload(
    trace: DebugTrace,
    *,
    snapshot_format: str = "svg",
    snapshot_directory: str = _DEFAULT_SNAPSHOT_DIRECTORY,
) -> dict[str, Any]:
    """Build a JSON-friendly browser payload for ``trace``."""
    entries = [
        build_replay_entry_payload(
            entry,
            snapshot_format=snapshot_format,
            snapshot_directory=snapshot_directory,
        )
        for entry in trace
    ]

    return {
        "entry_count": len(trace),
        "entries": entries,
    }


def build_replay_entry_payload(
    entry: DebugTimelineEntry,
    *,
    snapshot_format: str = "svg",
    snapshot_directory: str = _DEFAULT_SNAPSHOT_DIRECTORY,
) -> dict[str, int | str | bool | None]:
    """Build the browser payload for one timeline entry."""
    snapshot_file = None
    snapshot_metadata_file = None
    if entry.snapshot is not None:
        snapshot_file = _snapshot_relative_path(
            entry,
            snapshot_format=snapshot_format,
            snapshot_directory=snapshot_directory,
        )
        snapshot_metadata_file = _snapshot_metadata_relative_path(
            entry,
            snapshot_directory=snapshot_directory,
        )

    return {
        "index": entry.index,
        "event_type": entry.event.__class__.__name__,
        "event_summary": format_debug_event(entry.event),
        "has_snapshot": entry.snapshot is not None,
        "snapshot_file": snapshot_file,
        "snapshot_metadata_file": snapshot_metadata_file,
    }


def write_replay_payload(payload: dict[str, Any], path: str | Path) -> Path:
    """Write ``payload`` as readable UTF-8 JSON and return the output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def export_replay_bundle(
    trace: DebugTrace,
    directory: str | Path,
    *,
    snapshot_format: str = "svg",
) -> Path:
    """Export a static browser replay bundle for ``trace`` into ``directory``."""
    bundle_dir = Path(directory)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = bundle_dir / _DEFAULT_SNAPSHOT_DIRECTORY
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for entry in trace:
        if entry.snapshot is None:
            continue

        snapshot_path = snapshot_dir / trace_snapshot_filename(
            entry,
            format_str=snapshot_format,
        )
        export_snapshot_entry(
            entry.snapshot,
            snapshot_path,
            format_str=snapshot_format,
        )
        export_snapshot_json(
            entry.snapshot,
            snapshot_dir / trace_snapshot_metadata_filename(entry),
        )

    payload = build_replay_payload(trace, snapshot_format=snapshot_format)
    write_replay_payload(payload, bundle_dir / "trace.json")
    (bundle_dir / "index.html").write_text(
        render_replay_index_html(),
        encoding="utf-8",
    )
    return bundle_dir


def _snapshot_relative_path(
    entry: DebugTimelineEntry,
    *,
    snapshot_format: str,
    snapshot_directory: str,
) -> str:
    """Return the browser-relative snapshot path for ``entry``."""
    filename = trace_snapshot_filename(entry, format_str=snapshot_format)
    return PurePosixPath(snapshot_directory, filename).as_posix()


def _snapshot_metadata_relative_path(
    entry: DebugTimelineEntry,
    *,
    snapshot_directory: str,
) -> str:
    """Return the browser-relative snapshot metadata path for ``entry``."""
    filename = trace_snapshot_metadata_filename(entry)
    return PurePosixPath(snapshot_directory, filename).as_posix()


__all__ = [
    "build_replay_entry_payload",
    "build_replay_payload",
    "export_replay_bundle",
    "write_replay_payload",
]
