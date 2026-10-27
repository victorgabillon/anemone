"""Adapter that maps plain valanga dynamics to SearchDynamics."""

from dataclasses import dataclass

import valanga

from .search_dynamics import SearchDynamics


@dataclass(frozen=True)
class StatelessDynamicsAdapter[StateT: valanga.State](
    SearchDynamics[StateT, valanga.BranchKey]
):
    """Adapter that lets a plain valanga.Dynamics be used as SearchDynamics."""

    __anemone_search_dynamics__ = True

    dynamics: valanga.Dynamics[StateT]

    def legal_actions(
        self, state: StateT
    ) -> valanga.BranchKeyGeneratorP[valanga.BranchKey]:
        """Return legal actions from the wrapped valanga dynamics."""
        return self.dynamics.legal_actions(state)

    def step(
        self, state: StateT, action: valanga.BranchKey, *, depth: int
    ) -> valanga.Transition[StateT]:
        """Apply one action. The depth argument is accepted for protocol parity."""
        _ = depth
        return self.dynamics.step(state, action)

    def action_name(self, state: StateT, action: valanga.BranchKey) -> str:
        """Return a human-readable action name."""
        return self.dynamics.action_name(state, action)

    def action_from_name(self, state: StateT, name: str) -> valanga.BranchKey:
        """Parse an action from its name."""
        return self.dynamics.action_from_name(state, name)
