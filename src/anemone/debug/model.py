"""Immutable data models used by the debug tree tooling."""

from __future__ import annotations

from dataclasses import dataclass, field


def _empty_string_dict() -> dict[str, str]:
    """Return an empty string-keyed dictionary for dataclass defaults."""
    return {}


@dataclass(frozen=True)
class DebugNodeView:
    """Lightweight representation of a node for debugging/visualization."""

    node_id: str
    parent_ids: tuple[str, ...]
    depth: int
    label: str
    player_label: str | None = None
    state_tag: str | None = None
    direct_value: str | None = None
    backed_up_value: str | None = None
    principal_variation: str | None = None
    over_event: str | None = None
    index_fields: dict[str, str] = field(default_factory=_empty_string_dict)
    child_ids: tuple[str, ...] = ()
    edge_labels_by_child: dict[str, str] = field(default_factory=_empty_string_dict)


@dataclass(frozen=True)
class DebugEdgeView:
    """Directed connection between two debug nodes."""

    parent_id: str
    child_id: str
    label: str | None = None


@dataclass(frozen=True)
class DebugTreeSnapshot:
    """A complete snapshot of a search tree for visualization."""

    nodes: tuple[DebugNodeView, ...]
    root_id: str
    edges: tuple[DebugEdgeView, ...] = ()
