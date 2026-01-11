"""Defines state transition protocols and implementations for Anemone."""

from dataclasses import dataclass
from typing import Protocol

from valanga import BranchKey, State, StateModifications


class StateTransition[StateT](Protocol):
    """Protocol for state transitions in Anemone."""

    def copy_for_expansion(self, state: StateT, *, copy_stack: bool) -> StateT:
        """Return a copy of state suitable for expansion."""
        ...

    def step(
        self,
        state: StateT,
        *,
        branch_key: BranchKey,
    ) -> tuple[StateT, StateModifications | None]:
        """Apply a branch key to advance the state."""
        ...


@dataclass(frozen=True)
class ValangaStateTransition(StateTransition[State]):
    """State transition implementation using Valanga's State."""

    deep_copy_legal_moves: bool = False

    def copy_for_expansion(self, state: State, *, copy_stack: bool) -> State:
        """Copy the state, optionally copying its stack."""
        return state.copy(
            stack=copy_stack,
            deep_copy_legal_moves=self.deep_copy_legal_moves,
        )

    def step(
        self,
        state: State,
        *,
        branch_key: BranchKey,
    ) -> tuple[State, StateModifications | None]:
        """Advance state by a branch key and return modifications."""
        modifications = state.step(branch_key=branch_key)
        return state, modifications
