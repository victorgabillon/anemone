"""Single-agent objective implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .backup import BackupOp, MaxBackup
from .objective import BackupInput, ChildInfo, NodeKind, Objective, Role, StateT

if TYPE_CHECKING:
    from valanga import OverEvent


@dataclass(frozen=True, slots=True)
class SingleAgentObjective(Objective[StateT]):
    """Objective where a single role always chooses actions."""

    role: Role = "SOLO"
    backup_op: BackupOp = field(default_factory=MaxBackup)

    def node_kind(self, state: StateT) -> NodeKind:
        """Return decision-node semantics for all states."""
        _ = state
        return NodeKind.DECISION

    def to_play(self, state: StateT) -> Role:
        """Return the configured single role."""
        _ = state
        return self.role

    def child_sort_key(
        self,
        parent_state: StateT,
        child: ChildInfo,
    ) -> tuple[Any, ...]:
        """Prefer higher value, then shorter depth, then lower child id."""
        _ = parent_state
        return (-child.value, child.depth, child.child_id)

    def backup(self, parent_state: StateT, inp: BackupInput) -> float:
        """Back up with configured operator and optional prior fallback policy."""
        _ = parent_state
        if not inp.child_values:
            return inp.prior_value if inp.prior_value is not None else float("-inf")

        backed = self.backup_op(inp.child_values)
        if inp.prior_value is None:
            return backed
        return (
            max(backed, inp.prior_value) if not inp.all_children_generated else backed
        )

    def terminal_value(
        self,
        parent_state: StateT,
        over_event: OverEvent | None,
    ) -> float | None:
        """Leave terminal mapping undefined for this objective."""
        _ = parent_state
        _ = over_event
        return None
