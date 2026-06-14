"""Tests for rollout continuation action selectors."""

from random import Random
from typing import Any, cast

from anemone.rollouts import (
    FirstOpenableActionSelector,
    NoRolloutActionSelector,
    RandomOpenableActionSelector,
    RolloutDecisionContext,
)


def _context(openable_actions: tuple[int, ...]) -> RolloutDecisionContext[Any]:
    """Build a rollout decision context for selector tests."""
    return RolloutDecisionContext(
        current_node=cast("Any", object()),
        openable_actions=openable_actions,
        rollout_step_index=0,
    )


def test_first_openable_action_selector_preserves_openable_order() -> None:
    """First-openable selector returns the first openable action as provided."""
    selector: FirstOpenableActionSelector[Any] = FirstOpenableActionSelector()

    assert selector.choose_action(_context((2, 0, 1))) == 2


def test_no_rollout_action_selector_stops() -> None:
    """No-rollout selector returns no action."""
    selector: NoRolloutActionSelector[Any] = NoRolloutActionSelector()

    assert selector.choose_action(_context((2, 0, 1))) is None


def test_random_openable_action_selector_is_reproducible() -> None:
    """Random-openable selector uses its injected random generator."""
    selector_a: RandomOpenableActionSelector[Any] = RandomOpenableActionSelector(
        Random(123)
    )
    selector_b: RandomOpenableActionSelector[Any] = RandomOpenableActionSelector(
        Random(123)
    )
    context = _context((2, 0, 1))

    choices_a = [selector_a.choose_action(context) for _ in range(10)]
    choices_b = [selector_b.choose_action(context) for _ in range(10)]

    assert choices_a == choices_b
    assert all(choice in context.openable_actions for choice in choices_a)


def test_random_openable_action_selector_stops_when_no_action_is_openable() -> None:
    """Random-openable selector returns no action for an empty action set."""
    selector: RandomOpenableActionSelector[Any] = RandomOpenableActionSelector(
        Random(0)
    )

    assert selector.choose_action(_context(())) is None
