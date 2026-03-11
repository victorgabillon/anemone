"""Adapt runtime tree nodes into immutable debug snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .model import DebugNodeView, DebugTreeSnapshot

if TYPE_CHECKING:
    from anemone.nodes.itree_node import ITreeNode


class TreeSnapshotAdapter:
    """Build a ``DebugTreeSnapshot`` from a tree of ``ITreeNode`` instances."""

    def snapshot(self, root: ITreeNode) -> DebugTreeSnapshot:
        """Traverse the tree rooted at ``root`` and capture its current shape."""
        nodes: list[DebugNodeView] = []
        visited: set[str] = set()
        stack: list[ITreeNode] = [root]

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
            stack.extend(
                child for child in node.branches_children.values() if child is not None
            )

        return DebugTreeSnapshot(nodes=tuple(nodes), root_id=str(root.id))

    def _build_label(self, node: ITreeNode) -> str:
        """Build a simple textual label for a node."""
        return f"id={node.id}\ndepth={node.tree_depth}"
