"""Tests for live debug session bundle writing."""

# ruff: noqa: D103

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anemone.debug import (
    DebugNodeView,
    DebugTreeSnapshot,
    LiveDebugSessionRecorder,
    NodeSelected,
    SearchIterationStarted,
    render_replay_index_html,
)

if TYPE_CHECKING:
    from pathlib import Path


FIXED_SNAPSHOT = DebugTreeSnapshot(
    nodes=(
        DebugNodeView(
            node_id="1",
            parent_ids=(),
            depth=0,
            label="root",
            state_tag="root",
            direct_value="score=0.2",
            backed_up_value="score=0.9",
            principal_variation="a -> b",
            over_event="done",
            index_fields={"index": "2.0"},
        ),
    ),
    root_id="1",
)


def test_live_debug_session_recorder_initializes_session_directory(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "live-session"
    recorder = LiveDebugSessionRecorder(session_dir)

    assert recorder.to_trace().entries == ()
    assert session_dir.is_dir()
    assert (session_dir / "snapshots").is_dir()
    assert (session_dir / "index.html").exists()
    assert (session_dir / "session.json").exists()

    payload = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert payload["entry_count"] == 0
    assert payload["entries"] == []
    assert payload["is_live"] is True
    assert payload["is_complete"] is False


def test_live_debug_session_recorder_writes_events_incrementally(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "live-session"
    recorder = LiveDebugSessionRecorder(
        session_dir,
        snapshot_provider=lambda event: FIXED_SNAPSHOT,
        snapshot_policy=lambda event: isinstance(event, NodeSelected),
        snapshot_format="dot",
    )

    recorder.emit(SearchIterationStarted(iteration_index=0))
    first_payload = json.loads(
        (session_dir / "session.json").read_text(encoding="utf-8")
    )

    assert first_payload["entry_count"] == 1
    assert first_payload["entries"][0]["event_type"] == "SearchIterationStarted"
    assert first_payload["entries"][0]["event_fields"] == {"iteration_index": 0}
    assert first_payload["entries"][0]["breakpoint_hit"] is None
    assert first_payload["entries"][0]["snapshot_file"] is None
    assert first_payload["entries"][0]["snapshot_metadata_file"] is None

    recorder.emit(NodeSelected(node_id="9"))
    second_payload = json.loads(
        (session_dir / "session.json").read_text(encoding="utf-8")
    )

    assert second_payload["entry_count"] == 2
    assert second_payload["entries"][1]["event_summary"] == "node 9 selected"
    assert second_payload["entries"][1]["event_fields"] == {"node_id": "9"}
    assert second_payload["entries"][1]["breakpoint_hit"] is None
    assert second_payload["entries"][1]["snapshot_file"] == (
        "snapshots/0001_NodeSelected.dot"
    )
    assert second_payload["entries"][1]["snapshot_metadata_file"] == (
        "snapshots/0001_NodeSelected.snapshot.json"
    )


def test_live_debug_session_recorder_writes_snapshot_files_only_when_needed(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "live-session"
    snapshot_dir = session_dir / "snapshots"
    recorder = LiveDebugSessionRecorder(
        session_dir,
        snapshot_provider=lambda event: FIXED_SNAPSHOT,
        snapshot_policy=lambda event: isinstance(event, NodeSelected),
        snapshot_format="dot",
    )

    recorder.emit(SearchIterationStarted(iteration_index=0))
    assert tuple(snapshot_dir.iterdir()) == ()

    recorder.emit(NodeSelected(node_id="9"))

    snapshot_files = tuple(snapshot_dir.iterdir())
    assert len(snapshot_files) == 2
    assert {path.name for path in snapshot_files} == {
        "0001_NodeSelected.dot",
        "0001_NodeSelected.snapshot.json",
    }
    snapshot_payload = json.loads(
        (snapshot_dir / "0001_NodeSelected.snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot_payload["nodes"][0]["state_tag"] == "root"
    assert snapshot_payload["nodes"][0]["principal_variation"] == "a -> b"


def test_live_debug_session_recorder_finalize_marks_session_complete(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "live-session"
    recorder = LiveDebugSessionRecorder(session_dir)

    recorder.finalize()

    payload = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert payload["is_live"] is True
    assert payload["is_complete"] is True


def test_render_replay_index_html_contains_live_polling_hooks() -> None:
    html = render_replay_index_html()

    assert 'fetch("session.json"' in html
    assert 'fetch("trace.json"' in html
    assert "setInterval" in html
    assert 'id="auto-follow-latest"' in html
    assert 'id="timeline-search"' in html
    assert 'id="timeline-event-filter"' in html
    assert 'id="timeline-filter-status"' in html
    assert 'id="highlight-root-path"' in html
    assert 'id="highlight-neighborhood"' in html
    assert 'id="highlight-pv-path"' in html
    assert 'id="dim-unrelated-graph"' in html
    assert 'id="action-expand-node"' in html
    assert 'id="action-run-node-event"' in html
    assert 'id="action-run-node-value"' in html
    assert 'id="action-focus-node"' in html
    assert 'id="node-list"' in html
    assert 'id="node-details"' in html
