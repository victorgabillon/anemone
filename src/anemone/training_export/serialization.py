"""JSON-friendly serialization helpers for training tree snapshots."""

from __future__ import annotations

from typing import Any, cast

from .model import TrainingNodeSnapshot, TrainingTreeSnapshot


class MalformedTrainingTreeSnapshotError(TypeError):
    """Raised when training tree snapshot data is structurally malformed."""


class MalformedNodesFieldError(MalformedTrainingTreeSnapshotError):
    """Raised when the top-level nodes field is not a list of dictionaries."""

    def __init__(self) -> None:
        """Initialize with a stable explanation."""
        super().__init__(
            "Training tree snapshot `nodes` must be a list of objects."
        )


class MalformedNodeEntryError(MalformedTrainingTreeSnapshotError):
    """Raised when one node entry is not a dictionary."""

    def __init__(self) -> None:
        """Initialize with a stable explanation."""
        super().__init__(
            "Each training tree snapshot node must be a dictionary."
        )


def training_tree_snapshot_to_dict(
    snapshot: TrainingTreeSnapshot,
) -> dict[str, object]:
    """Serialize ``snapshot`` to a JSON-friendly dictionary."""
    return {
        "root_node_id": snapshot.root_node_id,
        "nodes": [_node_snapshot_to_dict(node) for node in snapshot.nodes],
        "created_at_unix_s": snapshot.created_at_unix_s,
        "metadata": dict(snapshot.metadata),
    }


def training_tree_snapshot_from_dict(
    data: dict[str, object],
) -> TrainingTreeSnapshot:
    """Deserialize a ``TrainingTreeSnapshot`` from JSON-friendly data."""
    nodes_data = data.get("nodes")
    if not isinstance(nodes_data, list):
        raise MalformedNodesFieldError
    return TrainingTreeSnapshot(
        root_node_id=_optional_str(data.get("root_node_id")),
        nodes=tuple(
            _node_snapshot_from_dict(_require_node_dict(item))
            for item in nodes_data
        ),
        created_at_unix_s=_optional_float(data.get("created_at_unix_s")),
        metadata=_metadata_dict(data.get("metadata")),
    )


def _node_snapshot_to_dict(snapshot: TrainingNodeSnapshot) -> dict[str, object]:
    """Serialize one node snapshot to JSON-friendly data."""
    return {
        "node_id": snapshot.node_id,
        "parent_ids": list(snapshot.parent_ids),
        "child_ids": list(snapshot.child_ids),
        "depth": snapshot.depth,
        "state_ref_payload": snapshot.state_ref_payload,
        "direct_value_scalar": snapshot.direct_value_scalar,
        "backed_up_value_scalar": snapshot.backed_up_value_scalar,
        "is_terminal": snapshot.is_terminal,
        "is_exact": snapshot.is_exact,
        "over_event_label": snapshot.over_event_label,
        "visit_count": snapshot.visit_count,
        "metadata": dict(snapshot.metadata),
    }


def _node_snapshot_from_dict(data: dict[str, object]) -> TrainingNodeSnapshot:
    """Deserialize one node snapshot from JSON-friendly data."""
    parent_ids = data.get("parent_ids", [])
    child_ids = data.get("child_ids", [])
    loaded_parent_ids = parent_ids if isinstance(parent_ids, list) else []
    loaded_child_ids = child_ids if isinstance(child_ids, list) else []
    return TrainingNodeSnapshot(
        node_id=str(data["node_id"]),
        parent_ids=tuple(str(parent_id) for parent_id in loaded_parent_ids),
        child_ids=tuple(str(child_id) for child_id in loaded_child_ids),
        depth=_coerce_int(data.get("depth", 0), default=0),
        state_ref_payload=data.get("state_ref_payload"),
        direct_value_scalar=_optional_float(data.get("direct_value_scalar")),
        backed_up_value_scalar=_optional_float(data.get("backed_up_value_scalar")),
        is_terminal=bool(data.get("is_terminal", False)),
        is_exact=bool(data.get("is_exact", False)),
        over_event_label=_optional_str(data.get("over_event_label")),
        visit_count=_optional_int(data.get("visit_count")),
        metadata=_metadata_dict(data.get("metadata")),
    )


def _require_node_dict(value: object) -> dict[str, object]:
    """Return one node payload or raise for malformed entries."""
    if not isinstance(value, dict):
        raise MalformedNodeEntryError
    return cast("dict[str, object]", value)


def _optional_str(value: object) -> str | None:
    """Return ``value`` as ``str`` unless it is ``None``."""
    return None if value is None else str(value)


def _optional_float(value: object) -> float | None:
    """Return ``value`` as ``float`` unless it is ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError


def _optional_int(value: object) -> int | None:
    """Return ``value`` as ``int`` unless it is ``None``."""
    if value is None:
        return None
    return _coerce_int(value)


def _metadata_dict(value: object) -> dict[str, Any]:
    """Return a shallow-copied metadata dictionary when possible."""
    if not isinstance(value, dict):
        return {}
    return dict(cast("dict[str, Any]", value))


def _coerce_int(value: object, *, default: int | None = None) -> int:
    """Return ``value`` as ``int`` for supported scalar payloads."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | str):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if default is not None:
        return default
    raise TypeError


__all__ = [
    "MalformedNodeEntryError",
    "MalformedNodesFieldError",
    "MalformedTrainingTreeSnapshotError",
    "training_tree_snapshot_from_dict",
    "training_tree_snapshot_to_dict",
]
