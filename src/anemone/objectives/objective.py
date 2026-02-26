"""Objective protocol and shared value-backup input types."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from valanga import OverEvent

StateT = TypeVar("StateT")
type Role = Hashable


class NodeKind(str, Enum):
    """Category of node handled by search."""

    DECISION = "decision"
    CHANCE = "chance"


@dataclass(frozen=True, slots=True)
class ChildInfo:
    """Data required to compare child nodes in a deterministic way."""

    value: float
    depth: int
    child_id: int


@dataclass(frozen=True, slots=True)
class BackupInput:
    """Inputs required by an objective to back up a parent value."""

    child_values: Sequence[float]
    prior_value: float | None
    all_children_generated: bool


@runtime_checkable
class Objective(Protocol[StateT]):
    """Protocol defining objective-specific search semantics."""

    def node_kind(self, state: StateT) -> NodeKind:
        """Return whether the node is a decision or chance node."""

    def to_play(self, state: StateT) -> Role | None:
        """Return role selecting actions at this node, if any."""

    def child_sort_key(self, parent_state: StateT, child: ChildInfo) -> tuple[Any, ...]:
        """Return sortable key used to rank child preference."""

    def backup(self, parent_state: StateT, inp: BackupInput) -> float:
        """Return backed-up value for parent node."""

    def terminal_value(
        self,
        parent_state: StateT,
        over_event: OverEvent | None,
    ) -> float | None:
        """Map terminal metadata to numeric value, if objective defines one."""
