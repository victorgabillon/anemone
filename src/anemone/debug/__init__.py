"""Utilities for capturing and rendering debug views of search trees."""

# pylint: disable=duplicate-code

from .breakpoints import (
    BackupFlagBreakpoint,
    DebugBreakpoint,
    EventTypeBreakpoint,
    IterationBreakpoint,
    NodeIdBreakpoint,
    any_breakpoint_matches,
    breakpoint_from_json,
    breakpoint_matches,
    breakpoint_to_json,
    breakpoints_from_json,
    breakpoints_to_json,
)
from .browser import (
    export_and_serve_trace,
    serve_live_debug_session,
    serve_replay_bundle,
)
from .dot_renderer import DotRenderer
from .evaluation_inspectors import EvaluationDebugInspectorResolver
from .events import (
    BackupFinished,
    BackupStarted,
    ChildLinked,
    DirectValueAssigned,
    NodeOpeningPlanned,
    NodeSelected,
    SearchDebugEvent,
    SearchIterationCompleted,
    SearchIterationStarted,
)
from .export import (
    export_snapshot_dot,
    export_snapshot_entry,
    export_snapshot_json,
    export_snapshot_render,
    export_trace_snapshots,
    export_trace_summary,
    render_snapshot,
)
from .html_templates import render_replay_index_html
from .label_builder import NodeDebugLabelBuilder
from .live_control import (
    ControlledTreeExploration,
    DebugCommand,
    DebugSessionController,
    write_debug_command,
)
from .live_session import LiveDebugSessionRecorder
from .live_setup import LiveDebugEnvironment, build_live_debug_environment
from .model import DebugEdgeView, DebugNodeView, DebugTreeSnapshot
from .observable import (
    NodeEvaluationSummary,
    ObservableAlgorithmNodeTreeManager,
    ObservableDirectEvaluator,
    ObservableNodeSelector,
    ObservableTreeExploration,
    ObservableUpdater,
    diff_new_children,
    snapshot_children,
    summarize_node_evaluation,
)
from .persistence import load_debug_trace, save_debug_trace
from .recording import (
    DebugTimelineEntry,
    DebugTimelineRecorder,
    DebugTrace,
    make_tree_snapshot_provider,
)
from .replay import TraceReplayView, format_debug_event
from .replay_bundle import (
    build_replay_entry_payload,
    build_replay_payload,
    export_replay_bundle,
    write_replay_payload,
)
from .sink import NullSearchDebugSink, SearchDebugSink
from .snapshot_adapter import TreeSnapshotAdapter
from .snapshot_serialization import (
    load_snapshot_json,
    snapshot_from_json,
    snapshot_to_json,
    write_snapshot_json,
)

__all__ = [
    "BackupFinished",
    "BackupFlagBreakpoint",
    "BackupStarted",
    "ChildLinked",
    "ControlledTreeExploration",
    "DebugBreakpoint",
    "DebugCommand",
    "DebugEdgeView",
    "DebugNodeView",
    "DebugSessionController",
    "DebugTimelineEntry",
    "DebugTimelineRecorder",
    "DebugTrace",
    "DebugTreeSnapshot",
    "DirectValueAssigned",
    "DotRenderer",
    "EvaluationDebugInspectorResolver",
    "EventTypeBreakpoint",
    "IterationBreakpoint",
    "LiveDebugEnvironment",
    "LiveDebugSessionRecorder",
    "NodeDebugLabelBuilder",
    "NodeEvaluationSummary",
    "NodeIdBreakpoint",
    "NodeOpeningPlanned",
    "NodeSelected",
    "NullSearchDebugSink",
    "ObservableAlgorithmNodeTreeManager",
    "ObservableDirectEvaluator",
    "ObservableNodeSelector",
    "ObservableTreeExploration",
    "ObservableUpdater",
    "SearchDebugEvent",
    "SearchDebugSink",
    "SearchIterationCompleted",
    "SearchIterationStarted",
    "TraceReplayView",
    "TreeSnapshotAdapter",
    "any_breakpoint_matches",
    "breakpoint_from_json",
    "breakpoint_matches",
    "breakpoint_to_json",
    "breakpoints_from_json",
    "breakpoints_to_json",
    "build_live_debug_environment",
    "build_replay_entry_payload",
    "build_replay_payload",
    "diff_new_children",
    "export_and_serve_trace",
    "export_replay_bundle",
    "export_snapshot_dot",
    "export_snapshot_entry",
    "export_snapshot_json",
    "export_snapshot_render",
    "export_trace_snapshots",
    "export_trace_summary",
    "format_debug_event",
    "load_debug_trace",
    "load_snapshot_json",
    "make_tree_snapshot_provider",
    "render_replay_index_html",
    "render_snapshot",
    "save_debug_trace",
    "serve_live_debug_session",
    "serve_replay_bundle",
    "snapshot_children",
    "snapshot_from_json",
    "snapshot_to_json",
    "summarize_node_evaluation",
    "write_debug_command",
    "write_replay_payload",
    "write_snapshot_json",
]
