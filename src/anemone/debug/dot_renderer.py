"""Render debug tree snapshots as Graphviz DOT graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from graphviz import Digraph

if TYPE_CHECKING:
    from .model import DebugTreeSnapshot


class _GraphvizNode(Protocol):
    def __call__(
        self,
        name: str,
        label: str | None = None,
        **attrs: str,
    ) -> None: ...


class _GraphvizEdge(Protocol):
    def __call__(
        self,
        tail_name: str,
        head_name: str,
        label: str | None = None,
        **attrs: str,
    ) -> None: ...


class _GraphvizDigraph(Protocol):
    node: _GraphvizNode
    edge: _GraphvizEdge


class DotRenderer:
    """Convert ``DebugTreeSnapshot`` instances into Graphviz DOT graphs."""

    def render(
        self,
        snapshot: DebugTreeSnapshot,
        format_str: str | None = None,
    ) -> Digraph:
        """Build a directed graph representation for the provided snapshot."""
        graph = Digraph() if format_str is None else Digraph(format=format_str)
        typed_graph = cast("_GraphvizDigraph", graph)
        add_node = typed_graph.node
        add_edge = typed_graph.edge

        for node in snapshot.nodes:
            add_node(node.node_id, label=node.label)

        if snapshot.edges:
            for edge in snapshot.edges:
                add_edge(edge.parent_id, edge.child_id, label=edge.label)
        else:
            for node in snapshot.nodes:
                for parent_id in node.parent_ids:
                    add_edge(parent_id, node.node_id)

        return graph
