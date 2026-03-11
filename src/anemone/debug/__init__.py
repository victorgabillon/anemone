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
from .sink import NullSearchDebugSink, SearchDebugSink
from .snapshot_adapter import TreeSnapshotAdapter

__all__ = [
    "BackupFinished",
    "BackupStarted",
    "ChildLinked",
    "DebugEdgeView",
    "DebugNodeView",
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
    "snapshot_children",
    "summarize_node_evaluation",
]
