"""Adapter that maps plain valanga dynamics to SearchDynamics."""

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

import valanga

from .search_dynamics import SearchDynamics

StateT = TypeVar("StateT", bound=valanga.State)
ActionT = TypeVar("ActionT", bound=Hashable)


@dataclass(frozen=True)
class StatelessDynamicsAdapter(SearchDynamics[StateT, ActionT]):
    """Adapter that lets a plain valanga.Dynamics be used as SearchDynamics."""

    __anemone_search_dynamics__ = True

    dynamics: valanga.Dynamics[StateT]

    def legal_actions(self, state: StateT) -> valanga.BranchKeyGeneratorP[ActionT]:
        """Return legal actions from the wrapped valanga dynamics."""
        return self.dynamics.legal_actions(state)  # type: ignore[return-value]

    def step(
        self, state: StateT, action: ActionT, *, depth: int
    ) -> valanga.Transition[StateT]:
        """Apply one action. The depth argument is accepted for protocol parity."""
        _ = depth
        return self.dynamics.step(state, action)

    def action_name(self, state: StateT, action: ActionT) -> str:
        """Return a human-readable action name."""
        return self.dynamics.action_name(state, action)

    def action_from_name(self, state: StateT, name: str) -> ActionT:
        """Parse an action from its name."""
        return self.dynamics.action_from_name(state, name)  # type: ignore[return-value]
