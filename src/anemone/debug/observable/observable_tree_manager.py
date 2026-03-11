"""Tree-manager wrappers emitting debug events without touching core code."""

from __future__ import annotations

from typing import Any

from anemone.debug.events import (
    BackupFinished,
    BackupStarted,
    ChildLinked,
    NodeOpeningPlanned,
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


class ObservableAlgorithmNodeTreeManager:
    """Wrap an algorithm tree manager and emit best-effort structural/debug events."""

    def __init__(self, base: Any, debug_sink: SearchDebugSink | None = None) -> None:
        """Store the wrapped tree manager and the sink receiving observed events."""
        self._base = base
        self._debug_sink = debug_sink or NullSearchDebugSink()

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

        self._emit_node_opening_planned_events(opening_instructions)
        result = self._base.open_instructions(
            tree=tree,
            opening_instructions=opening_instructions,
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
