"""Turn-driven two-player zero-sum objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .backup import BackupOp, MaxBackup, MinBackup
from .objective import BackupInput, ChildInfo, NodeKind, Objective, Role

if TYPE_CHECKING:
    from valanga import OverEvent


class HasTurn(Protocol):
    """Protocol for states exposing a ``turn`` attribute."""

    turn: Any


@dataclass(frozen=True, slots=True)
class ZeroSumTurnObjective(Objective[HasTurn]):
    """Two-player objective evaluated in the fixed ``max_role`` frame."""

    max_role: Role
    min_role: Role
    backup_max: BackupOp = field(default_factory=MaxBackup)
    backup_min: BackupOp = field(default_factory=MinBackup)

    def node_kind(self, state: HasTurn) -> NodeKind:
        """Return decision-node semantics for all states."""
        _ = state
        return NodeKind.DECISION

    def to_play(self, state: HasTurn) -> Role:
        """Map ``state.turn`` to either max or min role."""
        return self.max_role if state.turn == self.max_role else self.min_role

    def child_sort_key(
        self,
        parent_state: HasTurn,
        child: ChildInfo,
    ) -> tuple[Any, ...]:
        """Sort children according to role to play at parent."""
        if self.to_play(parent_state) == self.max_role:
            return (-child.value, child.depth, child.child_id)
        return (child.value, child.depth, child.child_id)

    def backup(self, parent_state: HasTurn, inp: BackupInput) -> float:
        """Back up via max/min operator with optional prior fallback policy."""
        if not inp.child_values:
            return inp.prior_value if inp.prior_value is not None else float("-inf")

        if self.to_play(parent_state) == self.max_role:
            backed = self.backup_max(inp.child_values)
            if inp.prior_value is None:
                return backed
            return (
                max(backed, inp.prior_value)
                if not inp.all_children_generated
                else backed
            )

        backed = self.backup_min(inp.child_values)
        if inp.prior_value is None:
            return backed
        return (
            min(backed, inp.prior_value) if not inp.all_children_generated else backed
        )

    def terminal_value(
        self,
        parent_state: HasTurn,
        over_event: OverEvent | None,
    ) -> float | None:
        """Leave terminal mapping undefined for this objective."""
        _ = parent_state
        _ = over_event
        return None
