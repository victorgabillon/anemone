"""Tests for trace replay export helpers."""

# ruff: noqa: D103

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.debug import (
    ChildLinked,
    DebugNodeView,
    DebugTimelineEntry,
    DebugTrace,
    DebugTreeSnapshot,
    NodeSelected,
    export_snapshot_dot,
    export_trace_snapshots,
    export_trace_summary,
)

if TYPE_CHECKING:
    from pathlib import Path


def _snapshot(root_id: str) -> DebugTreeSnapshot:
    return DebugTreeSnapshot(
        nodes=(
            DebugNodeView(
                node_id=root_id, parent_ids=(), depth=0, label=f"id={root_id}"
            ),
        ),
        root_id=root_id,
    )


def _trace_with_snapshots() -> DebugTrace:
    return DebugTrace(
        entries=(
            DebugTimelineEntry(index=0, event=NodeSelected(node_id="1")),
            DebugTimelineEntry(
                index=1,
                event=ChildLinked(
                    parent_id="1",
                    child_id="2",
                    branch_key="a",
                    was_already_present=False,
                ),
                snapshot=_snapshot("2"),
            ),
            DebugTimelineEntry(index=2, event=NodeSelected(node_id="2")),
            DebugTimelineEntry(
                index=3,
                event=NodeSelected(node_id="3"),
                snapshot=_snapshot("3"),
            ),
        )
    )


def test_export_snapshot_dot_writes_dot_source(tmp_path: Path) -> None:
    target = tmp_path / "single.dot"

    output_path = export_snapshot_dot(_snapshot("9"), target)

    assert output_path == target
    content = target.read_text(encoding="utf-8")
    assert "digraph" in content
    assert '9 [label="id=9"]' in content


def test_export_trace_summary_writes_event_lines(tmp_path: Path) -> None:
    trace = _trace_with_snapshots()
    summary_path = tmp_path / "summary.txt"

    output_path = export_trace_summary(trace, summary_path)

    assert output_path == summary_path
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("[0000] node 1 selected")
    assert lines[1].startswith("[0001] linked 1 --a--> 2")
    assert lines[3].startswith("[0003] node 3 selected")


def test_export_trace_snapshots_exports_snapshot_entries_to_dot(tmp_path: Path) -> None:
    trace = _trace_with_snapshots()

    outputs = export_trace_snapshots(trace, tmp_path / "frames", format_str="dot")

    assert len(outputs) == 2
    assert outputs[0].name == "0001_ChildLinked.dot"
    assert outputs[1].name == "0003_NodeSelected.dot"
    for path in outputs:
        assert path.exists()
        assert "digraph" in path.read_text(encoding="utf-8")
