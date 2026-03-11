from __future__ import annotations

from graphviz import Digraph

from .model import DebugTreeSnapshot


class DotRenderer:
    """Convert ``DebugTreeSnapshot`` instances into Graphviz DOT graphs."""

    def render(self, snapshot: DebugTreeSnapshot) -> Digraph:
        graph = Digraph()

        for node in snapshot.nodes:
            graph.node(node.node_id, label=node.label)

        for node in snapshot.nodes:
            for parent_id in node.parent_ids:
                graph.edge(parent_id, node.node_id)

        return graph
