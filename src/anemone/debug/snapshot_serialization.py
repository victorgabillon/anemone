"""JSON serialization helpers for debug tree snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .model import DebugEdgeView, DebugNodeView, DebugTreeSnapshot


def snapshot_to_json(snapshot: DebugTreeSnapshot) -> dict[str, Any]:
    """Serialize ``snapshot`` to a JSON-friendly dictionary."""
    return {
        "root_id": snapshot.root_id,
        "nodes": [
            {
                "node_id": node.node_id,
                "parent_ids": list(node.parent_ids),
                "depth": node.depth,
                "label": node.label,
            }
            for node in snapshot.nodes
        ],
        "edges": [
            {
                "parent_id": edge.parent_id,
                "child_id": edge.child_id,
                "label": edge.label,
            }
            for edge in snapshot.edges
        ],
    }


def snapshot_from_json(data: dict[str, Any]) -> DebugTreeSnapshot:
    """Deserialize a ``DebugTreeSnapshot`` from a JSON-friendly dictionary."""
    root_id = data["root_id"]
    nodes_data = data.get("nodes", [])
    edges_data = data.get("edges", [])

    return DebugTreeSnapshot(
        nodes=tuple(
            _node_from_json(cast("dict[str, Any]", item)) for item in nodes_data
        ),
        root_id=str(root_id),
        edges=tuple(
            _edge_from_json(cast("dict[str, Any]", item)) for item in edges_data
        ),
    )


def write_snapshot_json(snapshot: DebugTreeSnapshot, path: str | Path) -> Path:
    """Write ``snapshot`` as readable UTF-8 JSON to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(snapshot_to_json(snapshot), indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def load_snapshot_json(path: str | Path) -> DebugTreeSnapshot:
    """Load a snapshot JSON file from ``path``."""
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    return snapshot_from_json(cast("dict[str, Any]", loaded))


def _node_from_json(data: dict[str, Any]) -> DebugNodeView:
    """Deserialize one ``DebugNodeView`` from JSON data."""
    parent_ids = data.get("parent_ids", [])
    return DebugNodeView(
        node_id=str(data["node_id"]),
        parent_ids=tuple(str(parent_id) for parent_id in parent_ids),
        depth=int(data["depth"]),
        label=str(data["label"]),
    )


def _edge_from_json(data: dict[str, Any]) -> DebugEdgeView:
    """Deserialize one ``DebugEdgeView`` from JSON data."""
    label = data.get("label")
    return DebugEdgeView(
        parent_id=str(data["parent_id"]),
        child_id=str(data["child_id"]),
        label=None if label is None else str(label),
    )


__all__ = [
    "load_snapshot_json",
    "snapshot_from_json",
    "snapshot_to_json",
    "write_snapshot_json",
]
