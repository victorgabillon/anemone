"""Field-level delta types for tree-evaluation node updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valanga.evaluations import Value


@dataclass(slots=True, frozen=True)
class FieldChange[T]:
    """Describe one field's old/new values."""

    old: T
    new: T


@dataclass(slots=True, frozen=True)
class NodeDelta[BranchKeyT]:
    """Describe the node-observable fields changed by one node-local update."""

    value: FieldChange[Value | None] | None = None
    pv_version: FieldChange[int] | None = None
    best_branch: FieldChange[BranchKeyT | None] | None = None
    all_branches_generated: FieldChange[bool] | None = None

    def is_empty(self) -> bool:
        """Return whether no observable node field changed."""
        return (
            self.value is None
            and self.pv_version is None
            and self.best_branch is None
            and self.all_branches_generated is None
        )
