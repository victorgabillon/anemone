"""Explicit generic state-handle abstractions used by core node types."""

from dataclasses import dataclass
from typing import Protocol

from valanga import State


class StateHandle[StateT: State = State](Protocol):
    """Minimal explicit handle for retrieving one node state."""

    def get(self) -> StateT:
        """Return the concrete state behind this handle."""
        ...


@dataclass(frozen=True, slots=True)
class MaterializedStateHandle[StateT: State = State]:
    """Handle that simply stores an already materialized state object."""

    state_: StateT

    def get(self) -> StateT:
        """Return the stored concrete state."""
        return self.state_
