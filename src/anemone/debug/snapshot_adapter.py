"""Adapt runtime tree nodes into immutable debug snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .label_builder import NodeDebugLabelBuilder
from .model import DebugEdgeView, DebugNodeView, DebugTreeSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable

    from valanga import BranchKey

    from anemone.nodes.itree_node import ITreeNode


class TreeSnapshotAdapter:
    """Build a ``DebugTreeSnapshot`` from a tree of ``ITreeNode`` instances."""

    def __init__(
        self,
        label_builder: NodeDebugLabelBuilder | None = None,
        edge_label_builder: (Callable[[Any, BranchKey, Any], str | None] | None) = None,
    ) -> None:
        """Initialize the adapter with optional node and edge formatters."""
        self._label_builder = label_builder or NodeDebugLabelBuilder()
        self._edge_label_builder = edge_label_builder

    def snapshot(self, root: ITreeNode[Any]) -> DebugTreeSnapshot:
        """Traverse the tree rooted at ``root`` and capture its current shape."""
        nodes: list[DebugNodeView] = []
        edges: list[DebugEdgeView] = []
        seen_edges: set[tuple[str, str, str | None]] = set()
        visited: set[str] = set()
        stack: list[ITreeNode[Any]] = [root]

        while stack:
            node = stack.pop()
            node_id = str(node.id)

            if node_id in visited:
                continue

            visited.add(node_id)
            parent_ids = tuple(str(parent.id) for parent in node.parent_nodes)
            label = self._build_label(node)

            nodes.append(
                DebugNodeView(
                    node_id=node_id,
                    parent_ids=parent_ids,
                    depth=node.tree_depth,
                    label=label,
                )
            )

            # Some trees may store unopened or absent children as ``None``.
            for branch_key, child in node.branches_children.items():
                if child is None:
                    continue

                edge_label = self._build_edge_label(node, branch_key, child)
                edge_key = (node_id, str(child.id), edge_label)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(
                        DebugEdgeView(
                            parent_id=node_id,
                            child_id=str(child.id),
                            label=edge_label,
                        )
                    )

                stack.append(child)

        return DebugTreeSnapshot(
            nodes=tuple(nodes),
            edges=tuple(edges),
            root_id=str(root.id),
        )

    def _build_label(self, node: ITreeNode[Any]) -> str:
        """Build a textual label for a node."""
        return self._label_builder.build_label(node)

    def _build_edge_label(
        self,
        parent: ITreeNode[Any],
        branch_key: BranchKey,
        child: ITreeNode[Any],
    ) -> str | None:
        """Build a textual label for an edge when configured."""
        if self._edge_label_builder is None:
            return None

        return self._edge_label_builder(parent, branch_key, child)
