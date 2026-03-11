"""Utilities for capturing and rendering debug views of search trees."""

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
from .label_builder import NodeDebugLabelBuilder
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
from .sink import NullSearchDebugSink, SearchDebugSink
from .snapshot_adapter import TreeSnapshotAdapter

__all__ = [
    "BackupFinished",
    "BackupStarted",
    "ChildLinked",
    "DebugEdgeView",
    "DebugNodeView",
    "DebugTimelineEntry",
    "DebugTimelineRecorder",
    "DebugTrace",
    "DebugTreeSnapshot",
    "DirectValueAssigned",
    "DotRenderer",
    "EvaluationDebugInspectorResolver",
    "NodeDebugLabelBuilder",
    "NodeEvaluationSummary",
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
    "TreeSnapshotAdapter",
    "diff_new_children",
    "load_debug_trace",
    "make_tree_snapshot_provider",
    "save_debug_trace",
    "snapshot_children",
    "summarize_node_evaluation",
]
