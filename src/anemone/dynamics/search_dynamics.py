"""Search-time dynamics interfaces for tree exploration."""

from typing import Protocol, TypeVar

import valanga

StateT = TypeVar("StateT")


class SearchDynamics(Protocol[StateT]):
    """Protocol for search-time rules and transitions."""

    def legal_actions(
        self, state: StateT
    ) -> valanga.BranchKeyGeneratorP[valanga.BranchKey]:
        """Return the legal actions for the given state."""
        ...

    def step(
        self, state: StateT, action: valanga.BranchKey, *, depth: int
    ) -> valanga.Transition[StateT]:
        """Perform a step in the dynamics, returning the resulting transition. The depth parameter can be used to implement depth-dependent dynamics, such as opening instructions or late-game heuristics."""
        ...

    def action_name(self, state: StateT, action: valanga.BranchKey) -> str:
        """Return the name of the action in the given state."""
        ...

    def action_from_name(self, state: StateT, name: str) -> valanga.BranchKey:
        """Return the action corresponding to the given name in the given state."""
        ...


class StatelessDynamicsAdapter(SearchDynamics[StateT]):
    """Adapter turning plain :class:`valanga.Dynamics` into SearchDynamics."""

    def __init__(self, dyn: valanga.Dynamics[StateT]) -> None:
        """Initialize the adapter with the given dynamics."""
        self._dyn = dyn

    def legal_actions(
        self, state: StateT
    ) -> valanga.BranchKeyGeneratorP[valanga.BranchKey]:
        """Return the legal actions for the given state."""
        return self._dyn.legal_actions(state)

    def step(
        self, state: StateT, action: valanga.BranchKey, *, depth: int
    ) -> valanga.Transition[StateT]:
        """Perform a step in the dynamics, returning the resulting transition. The depth parameter is ignored in this adapter."""
        return self._dyn.step(state, action)

    def action_name(self, state: StateT, action: valanga.BranchKey) -> str:
        """Return the name of the action in the given state."""
        return self._dyn.action_name(state, action)

    def action_from_name(self, state: StateT, name: str) -> valanga.BranchKey:
        """Return the action corresponding to the given name in the given state."""
        return self._dyn.action_from_name(state, name)
