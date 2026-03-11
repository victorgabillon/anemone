"""Structured domain events emitted during search exploration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchIterationStarted:
    """The search exploration is starting one iteration."""

    iteration_index: int


@dataclass(frozen=True, slots=True)
class NodeSelected:
    """A node has been selected for opening."""

    node_id: str


@dataclass(frozen=True, slots=True)
class NodeOpeningPlanned:
    """The search plans to open one or more branches from a node."""

    node_id: str
    branch_count: int


@dataclass(frozen=True, slots=True)
class ChildLinked:
    """A parent/branch/child relationship has been established in the tree."""

    parent_id: str
    child_id: str
    branch_key: str
    was_already_present: bool


@dataclass(frozen=True, slots=True)
class DirectValueAssigned:
    """A node received a direct evaluation value."""

    node_id: str
    value_repr: str


@dataclass(frozen=True, slots=True)
class BackupStarted:
    """Backup processing started for a node."""

    node_id: str


@dataclass(frozen=True, slots=True)
class BackupFinished:
    """Backup processing completed for a node."""

    node_id: str
    value_changed: bool
    pv_changed: bool
    over_changed: bool


@dataclass(frozen=True, slots=True)
class SearchIterationCompleted:
    """The search exploration finished one iteration."""

    iteration_index: int


type SearchDebugEvent = (
    SearchIterationStarted
    | NodeSelected
    | NodeOpeningPlanned
    | ChildLinked
    | DirectValueAssigned
    | BackupStarted
    | BackupFinished
    | SearchIterationCompleted
)


__all__ = [
    "BackupFinished",
    "BackupStarted",
    "ChildLinked",
    "DirectValueAssigned",
    "NodeOpeningPlanned",
    "NodeSelected",
    "SearchDebugEvent",
    "SearchIterationCompleted",
    "SearchIterationStarted",
]
