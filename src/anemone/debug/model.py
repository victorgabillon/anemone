"""Immutable data models used by the debug tree tooling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DebugNodeView:
    """Lightweight representation of a node for debugging/visualization."""

    node_id: str
    parent_ids: tuple[str, ...]
    depth: int
    label: str


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
