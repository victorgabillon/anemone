"""Selector wrapper emitting debug events without modifying core code."""

from __future__ import annotations

from typing import Any

from anemone.debug.events import NodeSelected
from anemone.debug.observable.state_diff import (
    collect_unique_nodes_from_opening_instructions,
)
from anemone.debug.sink import NullSearchDebugSink, SearchDebugSink


class ObservableNodeSelector:
    """Wrap a node selector and emit best-effort node-selection events."""

    def __init__(self, base: Any, debug_sink: SearchDebugSink | None = None) -> None:
        """Store the wrapped selector and the sink receiving selection events."""
        self._base = base
        self._debug_sink = debug_sink or NullSearchDebugSink()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped selector."""
        return getattr(self._base, name)

    def choose_node_and_branch_to_open(
        self,
        *,
        tree: Any,
        latest_tree_expansions: Any,
    ) -> Any:
        """Delegate node selection and emit one event per selected node."""
        opening_instructions = self._base.choose_node_and_branch_to_open(
            tree=tree,
            latest_tree_expansions=latest_tree_expansions,
        )

        for node in collect_unique_nodes_from_opening_instructions(
            opening_instructions
        ):
            node_id = str(node.id)
            self._debug_sink.emit(NodeSelected(node_id=node_id))

        return opening_instructions
