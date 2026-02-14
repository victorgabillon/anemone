"""Search-time dynamics interfaces for tree exploration."""

from __future__ import annotations

from typing import Protocol, TypeVar

import valanga

StateT = TypeVar("StateT")


class SearchDynamics(Protocol[StateT]):
    """Protocol for search-time rules and transitions."""

    def legal_actions(
        self, state: StateT
    ) -> valanga.BranchKeyGeneratorP[valanga.BranchKey]: ...

    def step(
        self, state: StateT, action: valanga.BranchKey, *, depth: int
    ) -> valanga.Transition[StateT]: ...

    def action_name(self, state: StateT, action: valanga.BranchKey) -> str: ...

    def action_from_name(self, state: StateT, name: str) -> valanga.BranchKey: ...


class StatelessDynamicsAdapter(SearchDynamics[StateT]):
    """Adapter turning plain :class:`valanga.Dynamics` into SearchDynamics."""

    def __init__(self, dyn: valanga.Dynamics[StateT]) -> None:
        self._dyn = dyn

    def legal_actions(
        self, state: StateT
    ) -> valanga.BranchKeyGeneratorP[valanga.BranchKey]:
        return self._dyn.legal_actions(state)

    def step(
        self, state: StateT, action: valanga.BranchKey, *, depth: int
    ) -> valanga.Transition[StateT]:
        return self._dyn.step(state, action)

    def action_name(self, state: StateT, action: valanga.BranchKey) -> str:
        return self._dyn.action_name(state, action)

    def action_from_name(self, state: StateT, name: str) -> valanga.BranchKey:
        return self._dyn.action_from_name(state, name)
