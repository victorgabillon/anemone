"""Search-time dynamics interfaces for tree exploration."""

from collections.abc import Hashable
from typing import Literal, Protocol, TypeVar

import valanga

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT", bound=Hashable)


class SearchDynamics(Protocol[StateT, ActionT]):
    """Protocol for search-time rules and transitions."""

    __anemone_search_dynamics__: Literal[True]

    def legal_actions(self, state: StateT) -> valanga.BranchKeyGeneratorP[ActionT]:
        """Return the legal actions for the given state."""
        ...

    def step(
        self, state: StateT, action: ActionT, *, depth: int
    ) -> valanga.Transition[StateT]:
        """Perform a step in the dynamics, returning the resulting transition. The depth parameter can be used to implement depth-dependent dynamics, such as opening instructions or late-game heuristics."""
        ...

    def action_name(self, state: StateT, action: ActionT) -> str:
        """Return the name of the action in the given state."""
        ...

    def action_from_name(self, state: StateT, name: str) -> ActionT:
        """Return the action corresponding to the given name in the given state."""
        ...
