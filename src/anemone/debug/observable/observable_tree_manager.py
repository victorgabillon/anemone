"""Tree-manager wrappers emitting debug events without touching core code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anemone.debug.events import (
    BackupFinished,
    BackupStarted,
    ChildLinked,
    NodeOpeningPlanned,
    SearchDebugEvent,
)
from anemone.debug.observable.observable_direct_evaluator import (
    ObservableDirectEvaluator,
)
from anemone.debug.observable.state_diff import (
    collect_nodes_and_ancestors,
    collect_nodes_from_tree_expansions,
    collect_opening_instructions,
    collect_unique_nodes_from_opening_instructions,
    diff_new_children,
    snapshot_children,
    summarize_node_evaluation,
)
from anemone.debug.sink import NullSearchDebugSink, SearchDebugSink


def _empty_debug_events() -> list[SearchDebugEvent]:
    """Return an empty typed buffer for replayable debug events."""
    return []


class ObservableAlgorithmNodeTreeManager:
    """Wrap an algorithm tree manager and emit best-effort structural/debug events."""

    def __init__(self, base: Any, debug_sink: SearchDebugSink | None = None) -> None:
        """Store the wrapped tree manager and the sink receiving observed events."""
        self._base = base
        self._debug_sink = debug_sink or NullSearchDebugSink()
        _install_observable_direct_evaluator(self._base, debug_sink=self._debug_sink)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped tree manager."""
        return getattr(self._base, name)

    @property
    def dynamics(self) -> Any:
        """Return the wrapped manager's search dynamics."""
        return self._base.dynamics

    def open_instructions(self, tree: Any, opening_instructions: Any) -> Any:
        """Delegate opening to the base manager and infer structural events."""
        nodes_to_open = collect_unique_nodes_from_opening_instructions(
            opening_instructions
        )
        before_children_by_node = {
            id(node): snapshot_children(node) for node in nodes_to_open
        }
        buffered_direct_value_events = _BufferedDebugSink()
        observable_direct_evaluator = _find_observable_direct_evaluator(self._base)
        original_direct_evaluator_sink: SearchDebugSink | None = None
        if observable_direct_evaluator is not None:
            original_direct_evaluator_sink = observable_direct_evaluator.debug_sink
            observable_direct_evaluator.set_debug_sink(buffered_direct_value_events)

        self._emit_node_opening_planned_events(opening_instructions)
        try:
            result = self._base.open_instructions(
                tree=tree,
                opening_instructions=opening_instructions,
            )
        finally:
            if observable_direct_evaluator is not None:
                assert original_direct_evaluator_sink is not None
                observable_direct_evaluator.set_debug_sink(
                    original_direct_evaluator_sink
                )

        for node in nodes_to_open:
            before = before_children_by_node[id(node)]
            after = snapshot_children(node)
            for branch_key, child_id in diff_new_children(before, after):
                self._debug_sink.emit(
                    ChildLinked(
                        parent_id=str(node.id),
                        child_id=child_id,
                        branch_key=branch_key,
                        was_already_present=branch_key in before,
                    )
                )

        buffered_direct_value_events.flush_to(self._debug_sink)
        return result

    def update_indices(self, tree: Any) -> None:
        """Delegate index updates to the wrapped tree manager."""
        self._base.update_indices(tree=tree)

    def update_backward(self, tree_expansions: Any) -> Any:
        """Delegate backward updates and infer best-effort backup events."""
        observed_nodes = collect_nodes_and_ancestors(
            collect_nodes_from_tree_expansions(tree_expansions)
        )
        before_summaries = {
            id(node): summarize_node_evaluation(node) for node in observed_nodes
        }

        for node in observed_nodes:
            self._debug_sink.emit(BackupStarted(node_id=str(node.id)))

        result = self._base.update_backward(tree_expansions=tree_expansions)

        for node in observed_nodes:
            before = before_summaries[id(node)]
            after = summarize_node_evaluation(node)
            self._debug_sink.emit(
                BackupFinished(
                    node_id=str(node.id),
                    value_changed=(
                        before.backed_up_value_repr != after.backed_up_value_repr
                    ),
                    pv_changed=before.pv_repr != after.pv_repr,
                    over_changed=before.over_repr != after.over_repr,
                )
            )

        return result

    def print_best_line(self, tree: Any) -> None:
        """Delegate best-line printing to the wrapped tree manager."""
        self._base.print_best_line(tree=tree)

    def _emit_node_opening_planned_events(self, opening_instructions: Any) -> None:
        """Emit one opening-planned event per node in the instruction batch."""
        branch_counts_by_node_id: dict[str, int] = {}
        for opening_instruction in collect_opening_instructions(opening_instructions):
            node_id = str(opening_instruction.node_to_open.id)
            branch_counts_by_node_id[node_id] = (
                branch_counts_by_node_id.get(node_id, 0) + 1
            )

        for node_id, branch_count in branch_counts_by_node_id.items():
            self._debug_sink.emit(
                NodeOpeningPlanned(node_id=node_id, branch_count=branch_count)
            )


@dataclass(slots=True)
class _BufferedDebugSink:
    """Collect debug events and replay them later in the same order."""

    events: list[SearchDebugEvent] = field(default_factory=_empty_debug_events)

    def emit(self, event: SearchDebugEvent) -> None:
        """Store one event for later replay."""
        self.events.append(event)

    def flush_to(self, sink: SearchDebugSink) -> None:
        """Emit all buffered events to ``sink`` in FIFO order."""
        for event in self.events:
            sink.emit(event)
        self.events.clear()


def _install_observable_direct_evaluator(
    tree_manager_like: Any,
    *,
    debug_sink: SearchDebugSink,
) -> None:
    """Wrap the underlying direct evaluator so direct-value events are emitted."""
    manager_with_evaluator = _find_manager_with_node_evaluator(tree_manager_like)
    if manager_with_evaluator is None:
        return

    node_evaluator = getattr(manager_with_evaluator, "node_evaluator", None)
    if node_evaluator is None or isinstance(node_evaluator, ObservableDirectEvaluator):
        return

    manager_with_evaluator.node_evaluator = ObservableDirectEvaluator(
        node_evaluator,
        debug_sink=debug_sink,
    )


def _find_manager_with_node_evaluator(tree_manager_like: Any) -> Any | None:
    """Return the first wrapped manager object that actually owns node_evaluator."""
    current = tree_manager_like
    seen_objects: set[int] = set()

    while current is not None and id(current) not in seen_objects:
        seen_objects.add(id(current))
        current_attributes = getattr(current, "__dict__", None)
        if (
            isinstance(current_attributes, dict)
            and "node_evaluator" in current_attributes
        ):
            return current
        current = getattr(current, "_base", None)

    return None


def _find_observable_direct_evaluator(
    tree_manager_like: Any,
) -> ObservableDirectEvaluator | None:
    """Return the installed observable direct evaluator when available."""
    manager_with_evaluator = _find_manager_with_node_evaluator(tree_manager_like)
    if manager_with_evaluator is None:
        return None

    node_evaluator = getattr(manager_with_evaluator, "node_evaluator", None)
    if isinstance(node_evaluator, ObservableDirectEvaluator):
        return node_evaluator
    return None
