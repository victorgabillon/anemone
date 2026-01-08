from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from valanga import BranchKey, State, StateModifications


class StateTransition[TState](Protocol):
    def copy_for_expansion(self, state: TState, *, copy_stack: bool) -> TState: ...

    def step(
        self,
        state: TState,
        *,
        branch_key: BranchKey,
    ) -> tuple[TState, StateModifications | None]: ...


@dataclass(frozen=True)
class ValangaStateTransition(StateTransition[State]):
    deep_copy_legal_moves: bool = False

    def copy_for_expansion(self, state: State, *, copy_stack: bool) -> State:
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
        modifications = state.step(branch_key=branch_key)
        return state, modifications
