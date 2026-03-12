"""Tests for browser replay bundle helpers."""

# ruff: noqa: D103

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anemone.debug import (
    BackupFinished,
    ChildLinked,
    DebugNodeView,
    DebugTimelineEntry,
    DebugTrace,
    DebugTreeSnapshot,
    DirectValueAssigned,
    NodeSelected,
    SearchIterationStarted,
    build_replay_payload,
    export_replay_bundle,
    render_replay_index_html,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_snapshot(node_id: str) -> DebugTreeSnapshot:
    return DebugTreeSnapshot(
        nodes=(
            DebugNodeView(
                node_id=node_id,
                parent_ids=(),
                depth=0,
                label=f"id={node_id}",
            ),
        ),
        root_id=node_id,
    )


def _build_trace() -> DebugTrace:
    return DebugTrace(
        entries=(
            DebugTimelineEntry(
                index=0,
                event=SearchIterationStarted(iteration_index=1),
            ),
            DebugTimelineEntry(index=1, event=NodeSelected(node_id="5")),
            DebugTimelineEntry(
                index=2,
                event=ChildLinked(
                    parent_id="5",
                    child_id="8",
                    branch_key="a",
                    was_already_present=False,
                ),
                snapshot=_make_snapshot("8"),
            ),
            DebugTimelineEntry(
                index=3,
                event=DirectValueAssigned(node_id="8", value_repr="score=0.5"),
            ),
            DebugTimelineEntry(
                index=4,
                event=BackupFinished(
                    node_id="5",
                    value_changed=True,
                    pv_changed=False,
                    over_changed=False,
                ),
                snapshot=_make_snapshot("5"),
            ),
        )
    )


def test_build_replay_payload_is_json_friendly() -> None:
    payload = build_replay_payload(_build_trace())

    assert payload["entry_count"] == 5
    json.dumps(payload)

    entries = payload["entries"]
    assert isinstance(entries, list)
    assert entries[0]["event_type"] == "SearchIterationStarted"
    assert entries[0]["event_summary"] == "iteration 1 started"
    assert entries[0]["has_snapshot"] is False
    assert entries[0]["snapshot_file"] is None
    assert entries[2]["snapshot_file"] == "snapshots/0002_ChildLinked.svg"
    assert entries[3]["snapshot_file"] is None
    assert entries[4]["snapshot_file"] == "snapshots/0004_BackupFinished.svg"


def test_export_replay_bundle_writes_expected_files(tmp_path: Path) -> None:
    bundle_dir = export_replay_bundle(
        _build_trace(),
        tmp_path / "bundle",
        snapshot_format="dot",
    )

    assert bundle_dir == tmp_path / "bundle"
    assert (bundle_dir / "index.html").exists()
    assert (bundle_dir / "trace.json").exists()
    assert (bundle_dir / "snapshots" / "0002_ChildLinked.dot").exists()
    assert (bundle_dir / "snapshots" / "0004_BackupFinished.dot").exists()


def test_export_replay_bundle_writes_expected_trace_json(tmp_path: Path) -> None:
    bundle_dir = export_replay_bundle(
        _build_trace(),
        tmp_path / "bundle",
        snapshot_format="dot",
    )

    payload = json.loads((bundle_dir / "trace.json").read_text(encoding="utf-8"))

    assert payload["entry_count"] == 5
    assert payload["entries"][2]["event_summary"] == "linked 5 --a--> 8"
    assert payload["entries"][2]["snapshot_file"] == "snapshots/0002_ChildLinked.dot"
    assert payload["entries"][3]["snapshot_file"] is None
    assert payload["entries"][4]["event_summary"].startswith("node 5 backup finished")


def test_render_replay_index_html_contains_expected_viewer_hooks() -> None:
    html = render_replay_index_html()

    assert 'fetch("trace.json")' in html
    assert 'id="timeline"' in html
    assert 'id="entry-summary"' in html
    assert 'id="snapshot-view"' in html
    assert "nearestSnapshotAtOrBefore" in html
