"""Tests for replay helpers built on top of ``DebugTrace``."""

# ruff: noqa: D103

from __future__ import annotations

from anemone.debug import (
    BackupFinished,
    ChildLinked,
    DebugNodeView,
    DebugTimelineEntry,
    DebugTrace,
    DebugTreeSnapshot,
    DirectValueAssigned,
    NodeOpeningPlanned,
    NodeSelected,
    SearchIterationStarted,
    TraceReplayView,
    format_debug_event,
)


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
                index=0, event=SearchIterationStarted(iteration_index=1)
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
                index=3, event=DirectValueAssigned(node_id="8", value_repr="score=0.5")
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


def test_trace_replay_view_navigation() -> None:
    replay = TraceReplayView(_build_trace())

    assert len(replay) == 5
    assert replay.entry_at(1).event == NodeSelected(node_id="5")


def test_trace_replay_view_snapshot_helpers() -> None:
    replay = TraceReplayView(_build_trace())

    assert replay.snapshot_at(1) is None
    assert replay.snapshot_at(2) == _make_snapshot("8")

    assert replay.nearest_snapshot_at_or_before(4) == _make_snapshot("5")
    assert replay.nearest_snapshot_at_or_before(3) == _make_snapshot("8")
    assert replay.nearest_snapshot_at_or_before(0) is None


def test_entries_with_snapshots_filters_trace_entries() -> None:
    replay = TraceReplayView(_build_trace())

    assert tuple(entry.index for entry in replay.entries_with_snapshots()) == (2, 4)


def test_format_debug_event_produces_readable_lines() -> None:
    assert (
        format_debug_event(SearchIterationStarted(iteration_index=3))
        == "iteration 3 started"
    )
    assert format_debug_event(NodeSelected(node_id="7")) == "node 7 selected"

    opening = format_debug_event(NodeOpeningPlanned(node_id="7", branch_count=2))
    assert "opening planned" in opening
    assert "2 branches" in opening

    linked = format_debug_event(
        ChildLinked(
            parent_id="17",
            child_id="23",
            branch_key="a",
            was_already_present=False,
        )
    )
    assert linked == "linked 17 --a--> 23"

    backup = format_debug_event(
        BackupFinished(
            node_id="17",
            value_changed=True,
            pv_changed=False,
            over_changed=False,
        )
    )
    assert "backup finished" in backup
    assert "value_changed=True" in backup
