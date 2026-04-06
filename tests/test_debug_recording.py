"""Tests for debug timeline recording and persistence."""

# ruff: noqa: D103
# pylint: disable=duplicate-code
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from anemone.debug import (
    DebugNodeView,
    DebugTimelineRecorder,
    DebugTreeSnapshot,
    NodeSelected,
    SearchIterationCompleted,
    SearchIterationStarted,
    load_debug_trace,
    make_tree_snapshot_provider,
    save_debug_trace,
)

if TYPE_CHECKING:
    from pathlib import Path


FIXED_SNAPSHOT = DebugTreeSnapshot(
    nodes=(DebugNodeView(node_id="1", parent_ids=(), depth=0, label="root"),),
    root_id="1",
)


@dataclass(eq=False)
class _FakeNode:
    id: int
    tree_depth: int
    state: Any | None = None
    tree_evaluation: Any | None = None
    branches_children: dict[str, _FakeNode | None] = field(default_factory=dict)
    parent_nodes: dict[_FakeNode, set[str]] = field(default_factory=dict)
    exploration_index_data: Any | None = None


def test_debug_timeline_recorder_stores_events_in_order() -> None:
    recorder = DebugTimelineRecorder()

    recorder.emit(SearchIterationStarted(iteration_index=0))
    recorder.emit(NodeSelected(node_id="7"))
    recorder.emit(SearchIterationCompleted(iteration_index=0))

    trace = recorder.to_trace()

    assert len(trace) == 3
    assert [entry.index for entry in trace] == [0, 1, 2]
    assert trace.entry_at(0).event == SearchIterationStarted(iteration_index=0)
    assert trace.entry_at(1).event == NodeSelected(node_id="7")
    assert trace.entry_at(2).event == SearchIterationCompleted(iteration_index=0)


def test_debug_timeline_recorder_captures_snapshots_when_configured() -> None:
    recorder = DebugTimelineRecorder(snapshot_provider=lambda event: FIXED_SNAPSHOT)

    recorder.emit(SearchIterationStarted(iteration_index=0))

    trace = recorder.to_trace()

    assert len(trace) == 1
    assert trace.entry_at(0).snapshot == FIXED_SNAPSHOT


def test_debug_timeline_recorder_snapshot_policy_filters_capture_points() -> None:
    recorder = DebugTimelineRecorder(
        snapshot_provider=lambda event: FIXED_SNAPSHOT,
        snapshot_policy=lambda event: isinstance(event, SearchIterationCompleted),
    )

    recorder.emit(SearchIterationStarted(iteration_index=0))
    recorder.emit(SearchIterationCompleted(iteration_index=0))

    trace = recorder.to_trace()

    assert trace.entry_at(0).snapshot is None
    assert trace.entry_at(1).snapshot == FIXED_SNAPSHOT


def test_debug_trace_replay_helpers_work() -> None:
    recorder = DebugTimelineRecorder(snapshot_provider=lambda event: FIXED_SNAPSHOT)
    recorder.emit(SearchIterationStarted(iteration_index=0))
    recorder.emit(NodeSelected(node_id="8"))
    recorder.emit(SearchIterationCompleted(iteration_index=0))

    trace = recorder.to_trace()

    assert trace.entry_at(1).event == NodeSelected(node_id="8")
    assert tuple(entry.event for entry in trace.events_of_type(NodeSelected)) == (
        NodeSelected(node_id="8"),
    )
    assert trace.last_snapshot() == FIXED_SNAPSHOT


def test_debug_trace_persistence_round_trip(tmp_path: Path) -> None:
    recorder = DebugTimelineRecorder(snapshot_provider=lambda event: FIXED_SNAPSHOT)
    recorder.emit(SearchIterationStarted(iteration_index=0))
    trace = recorder.to_trace()
    target_path = tmp_path / "trace.pkl"

    save_debug_trace(trace, target_path)
    loaded_trace = load_debug_trace(target_path)

    assert loaded_trace == trace


def test_make_tree_snapshot_provider_captures_current_root() -> None:
    root = _FakeNode(id=1, tree_depth=0)
    child = _FakeNode(
        id=2,
        tree_depth=1,
        parent_nodes={root: {"a"}},
    )
    root.branches_children["a"] = child

    snapshot_provider = make_tree_snapshot_provider(root_getter=lambda: root)

    snapshot = snapshot_provider(SearchIterationCompleted(iteration_index=0))

    assert snapshot is not None
    assert snapshot.root_id == "1"
    assert {node.node_id for node in snapshot.nodes} == {"1", "2"}
    assert snapshot.edges[0].parent_id == "1"
    assert snapshot.edges[0].child_id == "2"
    assert child.parent_nodes[root] == {"a"}
