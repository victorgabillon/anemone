"""Immutable data models for training-oriented tree export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TRAINING_TREE_SNAPSHOT_FORMAT_KIND = "training_tree_snapshot"
TRAINING_TREE_SNAPSHOT_FORMAT_VERSION = 1


def _empty_metadata() -> dict[str, Any]:
    """Return a typed empty metadata mapping."""
    return {}


@dataclass(frozen=True, slots=True)
class TrainingNodeSnapshot:
    """Machine-oriented snapshot of one search node."""

    node_id: str
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    depth: int
    state_ref_payload: Any | None
    direct_value_scalar: float | None
    backed_up_value_scalar: float | None
    is_terminal: bool
    is_exact: bool
    over_event_label: str | None = None
    visit_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class TrainingTreeSnapshot:
    """Machine-oriented snapshot of one exported search tree."""

    root_node_id: str | None
    nodes: tuple[TrainingNodeSnapshot, ...]
    created_at_unix_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=_empty_metadata)


__all__ = [
    "TRAINING_TREE_SNAPSHOT_FORMAT_KIND",
    "TRAINING_TREE_SNAPSHOT_FORMAT_VERSION",
    "TrainingNodeSnapshot",
    "TrainingTreeSnapshot",
]
