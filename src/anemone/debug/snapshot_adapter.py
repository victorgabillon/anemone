"""Adapt runtime tree nodes into immutable debug snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .formatting import safe_getattr
from .label_builder import NodeDebugLabelBuilder
from .model import DebugEdgeView, DebugNodeView, DebugTreeSnapshot
from .node_metadata_builder import NodeDebugMetadataBuilder

if TYPE_CHECKING:
    from collections.abc import Callable

    from anemone.nodes.itree_node import ITreeNode


class TreeSnapshotAdapter:
    """Build a ``DebugTreeSnapshot`` from a tree of ``ITreeNode`` instances."""

    def __init__(
        self,
        label_builder: NodeDebugLabelBuilder | None = None,
        metadata_builder: NodeDebugMetadataBuilder | None = None,
        edge_label_builder: (Callable[[Any, Any, Any], str | None] | None) = None,
    ) -> None:
        """Initialize the adapter with optional node and edge formatters."""
        self._label_builder = label_builder or NodeDebugLabelBuilder()
        resolved_metadata_builder = metadata_builder or cast(
            "NodeDebugMetadataBuilder | None",
            getattr(self._label_builder, "_metadata_builder", None),
        )
        self._metadata_builder = resolved_metadata_builder or NodeDebugMetadataBuilder()
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
            raw_parent_nodes = cast("object", safe_getattr(node, "parent_nodes") or {})
            parent_nodes = cast("dict[ITreeNode[Any], object]", raw_parent_nodes)
            parent_ids = tuple(str(parent.id) for parent in parent_nodes)
            label = self._build_label(node)
            metadata = self._metadata_builder.build_metadata(node)
            child_ids: list[str] = []
            edge_labels_by_child: dict[str, str] = {}

            # Some trees may store unopened or absent children as ``None``.
            raw_branches_children = cast(
                "object",
                safe_getattr(node, "branches_children") or {},
            )
            branches_children = cast(
                "dict[Any, ITreeNode[Any] | None]",
                raw_branches_children,
            )
            for branch_key, child in branches_children.items():
                if child is None:
                    continue

                child_id = str(child.id)
                child_ids.append(child_id)
                edge_label = self._build_edge_label(node, branch_key, child)
                if edge_label is not None:
                    edge_labels_by_child[child_id] = edge_label
                edge_key = (node_id, child_id, edge_label)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(
                        DebugEdgeView(
                            parent_id=node_id,
                            child_id=child_id,
                            label=edge_label,
                        )
                    )

                stack.append(child)

            nodes.append(
                DebugNodeView(
                    node_id=node_id,
                    parent_ids=parent_ids,
                    depth=node.tree_depth,
                    label=label,
                    state_tag=metadata.state_tag,
                    direct_value=metadata.direct_value,
                    backed_up_value=metadata.backed_up_value,
                    principal_variation=metadata.principal_variation,
                    over_event=metadata.over_event,
                    index_fields=metadata.index_fields,
                    child_ids=tuple(child_ids),
                    edge_labels_by_child=edge_labels_by_child,
                )
            )

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
        branch_key: Any,
        child: ITreeNode[Any],
    ) -> str | None:
        """Build a textual label for an edge when configured."""
        if self._edge_label_builder is None:
            return None

        return self._edge_label_builder(parent, branch_key, child)
