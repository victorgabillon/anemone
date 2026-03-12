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
                state_tag="root",
                direct_value="score=0.1",
                backed_up_value="score=0.8",
                principal_variation="a -> b",
                over_event="done",
                index_fields={"index": "1.5"},
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
                breakpoint_hit="bp-3",
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
    assert entries[0]["event_fields"] == {"iteration_index": 1}
    assert entries[0]["breakpoint_hit"] is None
    assert entries[0]["has_snapshot"] is False
    assert entries[0]["snapshot_file"] is None
    assert entries[0]["snapshot_metadata_file"] is None
    assert entries[2]["snapshot_file"] == "snapshots/0002_ChildLinked.svg"
    assert entries[2]["snapshot_metadata_file"] == (
        "snapshots/0002_ChildLinked.snapshot.json"
    )
    assert entries[2]["event_fields"] == {
        "parent_id": "5",
        "child_id": "8",
        "branch_key": "a",
        "was_already_present": False,
    }
    assert entries[3]["snapshot_file"] is None
    assert entries[3]["snapshot_metadata_file"] is None
    assert entries[3]["event_fields"] == {
        "node_id": "8",
        "value_repr": "score=0.5",
    }
    assert entries[4]["snapshot_file"] == "snapshots/0004_BackupFinished.svg"
    assert entries[4]["snapshot_metadata_file"] == (
        "snapshots/0004_BackupFinished.snapshot.json"
    )
    assert entries[4]["event_fields"] == {
        "node_id": "5",
        "value_changed": True,
        "pv_changed": False,
        "over_changed": False,
    }
    assert entries[4]["breakpoint_hit"] == "bp-3"


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
    assert (bundle_dir / "snapshots" / "0002_ChildLinked.snapshot.json").exists()
    assert (bundle_dir / "snapshots" / "0004_BackupFinished.dot").exists()
    assert (bundle_dir / "snapshots" / "0004_BackupFinished.snapshot.json").exists()
    snapshot_payload = json.loads(
        (bundle_dir / "snapshots" / "0002_ChildLinked.snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert snapshot_payload["nodes"][0]["state_tag"] == "root"
    assert snapshot_payload["nodes"][0]["direct_value"] == "score=0.1"
    assert snapshot_payload["nodes"][0]["index_fields"] == {"index": "1.5"}


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
    assert payload["entries"][2]["snapshot_metadata_file"] == (
        "snapshots/0002_ChildLinked.snapshot.json"
    )
    assert payload["entries"][2]["event_fields"]["branch_key"] == "a"
    assert payload["entries"][3]["snapshot_file"] is None
    assert payload["entries"][3]["snapshot_metadata_file"] is None
    assert payload["entries"][4]["event_summary"].startswith("node 5 backup finished")
    assert payload["entries"][4]["event_fields"]["value_changed"] is True
    assert payload["entries"][4]["breakpoint_hit"] == "bp-3"


def test_render_replay_index_html_contains_expected_viewer_hooks() -> None:
    html = render_replay_index_html()

    assert 'fetch("session.json"' in html
    assert 'fetch("trace.json"' in html
    assert 'fetch("control_state.json"' in html
    assert 'fetch("/command"' in html
    assert 'fetch("/breakpoints"' in html
    assert 'id="timeline"' in html
    assert 'id="timeline-search"' in html
    assert 'id="timeline-event-filter"' in html
    assert 'id="jump-entry-index"' in html
    assert 'id="jump-entry-button"' in html
    assert 'id="jump-next-breakpoint"' in html
    assert 'id="jump-next-pv-change"' in html
    assert 'id="jump-next-value-change"' in html
    assert 'id="jump-next-selected-node-event"' in html
    assert 'id="timeline-filter-status"' in html
    assert 'id="entry-summary"' in html
    assert 'id="snapshot-view"' in html
    assert 'id="highlight-root-path"' in html
    assert 'id="highlight-neighborhood"' in html
    assert 'id="highlight-pv-path"' in html
    assert 'id="dim-unrelated-graph"' in html
    assert 'id="node-list"' in html
    assert 'id="node-details"' in html
    assert 'id="selected-node-id"' in html
    assert 'id="selected-node-direct-value"' in html
    assert 'id="selected-node-index-fields"' in html
    assert 'id="selected-node-raw-label"' in html
    assert 'id="action-jump-node-event"' in html
    assert 'id="action-jump-node-value"' in html
    assert 'id="action-expand-node"' in html
    assert 'id="action-run-node-event"' in html
    assert 'id="action-run-node-value"' in html
    assert 'id="action-focus-node"' in html
    assert 'id="action-clear-node-focus"' in html
    assert "Pause" in html
    assert "Resume" in html
    assert "Step" in html
    assert "Clear Breakpoints" in html
    assert "nearestSnapshotAtOrBefore" in html
    assert "snapshot_metadata_file" in html
    assert "loadSnapshotMetadataForEntry" in html
    assert "renderNodeInspector" in html
    assert "renderListItems" in html
    assert "renderNodeDetails" in html
    assert "computeRootPathNodeIds" in html
    assert "computeRootPathEdgeKeys" in html
    assert "computeNeighborhoodNodeIds" in html
    assert "computeNeighborhoodEdgeKeys" in html
    assert "computePrincipalVariationNodeIds" in html
    assert "computePrincipalVariationEdgeKeys" in html
    assert "computeGraphFocusState" in html
    assert "renderGraphFocus" in html
    assert "extractEdgeKeyFromGraphvizGroup" in html
    assert "currentGraphEdgeMap" in html
    assert "edgeKey(" in html
    assert "entryMatchesSearch" in html
    assert "entryMatchesEventTypeFilter" in html
    assert "computeVisibleEntries" in html
    assert "jumpToEntryIndex" in html
    assert "jumpToNextBreakpointHit" in html
    assert "jumpToNextPvChange" in html
    assert "jumpToNextValueChange" in html
    assert "jumpToNextSelectedNodeEvent" in html
    assert "jumpToNextSelectedNodeValueChange" in html
    assert "focusTimelineOnNode" in html
    assert "clearTimelineNodeFocus" in html
    assert "event_fields" in html
    assert "breakpoint_hit" in html
    assert "focusedNodeId" in html
    assert "loadInlineSvg" in html
    assert "renderInlineSvg" in html
    assert "initializeGraphInteraction" in html
    assert "extractNodeIdFromGraphvizGroup" in html
    assert "renderGraphSelection" in html
    assert "selectNode" in html
    assert 'sendCommand("expand_node"' in html
    assert 'sendCommand("run_until_node_event"' in html
    assert 'sendCommand("run_until_node_value_change"' in html
    assert 'sendCommand("focus_node_timeline"' in html
    assert 'sendCommand("clear_timeline_focus"' in html
    assert "g.node" in html
    assert "g.edge" in html
    assert "is-root-path" in html
    assert "is-neighborhood" in html
    assert "is-pv-path" in html
    assert "is-dimmed" in html
    assert "is-selected" in html
