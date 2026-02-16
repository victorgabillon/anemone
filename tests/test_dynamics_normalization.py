"""Tests for search dynamics normalization."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import valanga

from anemone.dynamics import (
    SearchDynamics,
    StatelessDynamicsAdapter,
    normalize_search_dynamics,
)


@dataclass
class DummyState:
    """Minimal state used to exercise adapter behavior."""

    value: int


class DummyBranchKeyGenerator:
    """Minimal branch-key generator used by the fake dynamics."""

    sort_branch_keys: bool = False

    def __init__(self, keys: Sequence[valanga.BranchKey]) -> None:
        """Store available branch keys."""
        self._keys = tuple(keys)

    def get_all(self) -> Sequence[valanga.BranchKey]:
        """Return all branch keys."""
        return self._keys


class PlainDynamics:
    """Simple stateless dynamics with no depth-aware step."""

    def legal_actions(
        self, state: DummyState
    ) -> valanga.BranchKeyGeneratorP[valanga.BranchKey]:
        """Return one legal action."""
        _ = state
        return DummyBranchKeyGenerator((1,))

    def step(
        self, state: DummyState, action: valanga.BranchKey
    ) -> valanga.Transition[DummyState]:
        """Apply an action and return transition output."""
        next_state = DummyState(state.value + int(action))
        return valanga.Transition(
            next_state=next_state,
            modifications=None,
            is_over=False,
            over_event=None,
            info={},
        )

    def action_name(self, state: DummyState, action: valanga.BranchKey) -> str:
        """Serialize an action."""
        _ = state
        return str(action)

    def action_from_name(self, state: DummyState, name: str) -> valanga.BranchKey:
        """Deserialize an action."""
        _ = state
        return int(name)


class DepthDynamics(SearchDynamics[DummyState, Any]):
    """Search dynamics that already supports depth."""

    __anemone_search_dynamics__ = True

    def legal_actions(
        self, state: DummyState
    ) -> valanga.BranchKeyGeneratorP[valanga.BranchKey]:
        """Return one legal action."""
        _ = state
        return DummyBranchKeyGenerator((1,))

    def step(
        self, state: DummyState, action: valanga.BranchKey, *, depth: int
    ) -> valanga.Transition[DummyState]:
        """Apply an action with depth-dependent transition output."""
        next_state = DummyState(state.value + int(action) + depth)
        return valanga.Transition(
            next_state=next_state,
            modifications=None,
            is_over=False,
            over_event=None,
            info={},
        )

    def action_name(self, state: DummyState, action: valanga.BranchKey) -> str:
        """Serialize an action."""
        _ = state
        return str(action)

    def action_from_name(self, state: DummyState, name: str) -> valanga.BranchKey:
        """Deserialize an action."""
        _ = state
        return int(name)


def test_normalize_search_dynamics_returns_existing_search_dynamics() -> None:
    """Normalization should not wrap objects that already satisfy SearchDynamics."""
    dynamics = DepthDynamics()

    normalized = normalize_search_dynamics(dynamics)

    assert normalized is dynamics


def test_normalize_search_dynamics_wraps_plain_dynamics() -> None:
    """Normalization should adapt plain valanga-style dynamics to SearchDynamics."""
    dynamics = PlainDynamics()

    normalized = normalize_search_dynamics(dynamics)

    assert isinstance(normalized, StatelessDynamicsAdapter)
    transition = normalized.step(DummyState(2), 3, depth=99)
    assert transition.next_state == DummyState(5)
    assert transition.modifications is None
    assert transition.is_over is False
    assert transition.over_event is None
