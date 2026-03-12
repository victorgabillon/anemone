"""Export helpers for replaying debug traces and snapshots."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from .dot_renderer import DotRenderer
from .replay import format_debug_event

if TYPE_CHECKING:
    from graphviz import Digraph

    from .model import DebugTreeSnapshot
    from .recording import DebugTrace


_EVENT_NAME_SANITIZER = re.compile(r"[^0-9A-Za-z_]+")


def render_snapshot(
    snapshot: DebugTreeSnapshot,
    *,
    format_str: str | None = None,
) -> Digraph:
    """Render a debug snapshot as a ``graphviz.Digraph``."""
    return DotRenderer().render(snapshot, format_str=format_str)


def export_snapshot_dot(snapshot: DebugTreeSnapshot, path: str | Path) -> Path:
    """Write snapshot DOT source to ``path`` and return the output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    graph = render_snapshot(snapshot)
    target.write_text(graph.source, encoding="utf-8")
    return target


def export_snapshot_render(
    snapshot: DebugTreeSnapshot,
    path: str | Path,
    *,
    format_str: str = "svg",
) -> Path:
    """Render snapshot with Graphviz and return the resulting output path.

    The final file extension is driven by ``format_str``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    graph = render_snapshot(snapshot, format_str=format_str)
    rendered_path = graph.render(
        filename=target.stem,
        directory=str(target.parent),
        cleanup=True,
        format=format_str,
    )
    return Path(rendered_path)


def export_snapshot_entry(
    snapshot: DebugTreeSnapshot,
    path: str | Path,
    *,
    format_str: str = "svg",
) -> Path:
    """Export one snapshot, choosing DOT or rendered output by extension/format."""
    target = Path(path)
    if target.suffix.lower() == ".dot" or format_str.lower() == "dot":
        return export_snapshot_dot(snapshot, target)
    return export_snapshot_render(snapshot, target, format_str=format_str)


def export_trace_snapshots(
    trace: DebugTrace,
    directory: str | Path,
    *,
    format_str: str = "svg",
) -> tuple[Path, ...]:
    """Export snapshot-bearing entries to ``directory`` and return created paths."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []

    for entry in trace:
        if entry.snapshot is None:
            continue

        event_name = _safe_event_name(entry.event.__class__.__name__)
        extension = "dot" if format_str.lower() == "dot" else format_str
        output_path = output_dir / f"{entry.index:04d}_{event_name}.{extension}"
        exported.append(
            export_snapshot_entry(entry.snapshot, output_path, format_str=format_str)
        )

    return tuple(exported)


def export_trace_summary(trace: DebugTrace, path: str | Path) -> Path:
    """Write a readable timeline summary text file for ``trace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{entry.index:04d}] {format_debug_event(entry.event)}"
        for entry in trace.entries
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _safe_event_name(event_name: str) -> str:
    """Normalize event class names to deterministic filesystem-safe names."""
    sanitized = _EVENT_NAME_SANITIZER.sub("_", event_name).strip("_")
    return sanitized or "event"


__all__ = [
    "export_snapshot_dot",
    "export_snapshot_entry",
    "export_snapshot_render",
    "export_trace_snapshots",
    "export_trace_summary",
    "render_snapshot",
]
