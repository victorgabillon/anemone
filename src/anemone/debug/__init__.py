"""Public entrypoints for Anemone's live and replayable debug GUI.

The recommended way to run the browser debugger around a real exploration is
``build_live_debug_environment(...)``. Use ``serve_live_debug_session(...)`` to
view an on-disk live session while search is running, or ``serve_replay_bundle``
/ ``export_and_serve_trace(...)`` for offline replay workflows.
"""

# pylint: disable=duplicate-code
# ruff: noqa: RUF022

# Primary public entrypoints
# Advanced live control surface
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
    serve_debug_browser,
    serve_live_debug_session,
    serve_replay_bundle,
)

# Low-level debugging primitives
from .dot_renderer import DotRenderer
from .evaluation_inspectors import EvaluationDebugInspectorResolver

# Browser-visible event model
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

# Replay and trace helpers
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
    # Primary public entrypoints
    "LiveDebugEnvironment",
    "build_live_debug_environment",
    "serve_debug_browser",
    "serve_live_debug_session",
    "serve_replay_bundle",
    "export_and_serve_trace",
    # Replay and export helpers
    "DebugTimelineEntry",
    "DebugTimelineRecorder",
    "DebugTrace",
    "TraceReplayView",
    "build_replay_entry_payload",
    "build_replay_payload",
    "export_replay_bundle",
    "format_debug_event",
    "load_debug_trace",
    "make_tree_snapshot_provider",
    "save_debug_trace",
    "write_replay_payload",
    # Advanced live control surface
    "BackupFlagBreakpoint",
    "ControlledTreeExploration",
    "DebugBreakpoint",
    "DebugCommand",
    "DebugSessionController",
    "EventTypeBreakpoint",
    "IterationBreakpoint",
    "LiveDebugSessionRecorder",
    "NodeIdBreakpoint",
    "any_breakpoint_matches",
    "breakpoint_from_json",
    "breakpoint_matches",
    "breakpoint_to_json",
    "breakpoints_from_json",
    "breakpoints_to_json",
    "write_debug_command",
    # Browser-visible event model
    "BackupFinished",
    "BackupStarted",
    "ChildLinked",
    "DirectValueAssigned",
    "NodeOpeningPlanned",
    "NodeSelected",
    "SearchDebugEvent",
    "SearchIterationCompleted",
    "SearchIterationStarted",
    # Low-level debugging primitives
    "DebugEdgeView",
    "DebugNodeView",
    "DebugTreeSnapshot",
    "DotRenderer",
    "EvaluationDebugInspectorResolver",
    "NodeDebugLabelBuilder",
    "NodeEvaluationSummary",
    "NullSearchDebugSink",
    "ObservableAlgorithmNodeTreeManager",
    "ObservableDirectEvaluator",
    "ObservableNodeSelector",
    "ObservableTreeExploration",
    "ObservableUpdater",
    "SearchDebugSink",
    "TreeSnapshotAdapter",
    "diff_new_children",
    "export_snapshot_dot",
    "export_snapshot_entry",
    "export_snapshot_json",
    "export_snapshot_render",
    "export_trace_snapshots",
    "export_trace_summary",
    "load_snapshot_json",
    "render_replay_index_html",
    "render_snapshot",
    "snapshot_children",
    "snapshot_from_json",
    "snapshot_to_json",
    "summarize_node_evaluation",
    "write_snapshot_json",
]
