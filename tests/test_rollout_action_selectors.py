"""Tests for rollout continuation action selectors."""

from random import Random
from typing import Any, cast

from anemone.rollouts import (
    FirstOpenableActionSelector,
    NoRolloutActionSelector,
    RandomOpenableActionSelector,
    RolloutDecisionContext,
)


def _context(
    *,
    legal_actions: tuple[int, ...] = (2, 0, 1),
    opened_actions: tuple[int, ...] = (),
    openable_actions: tuple[int, ...] = (2, 0, 1),
) -> RolloutDecisionContext[Any]:
    """Build a rollout decision context for selector tests."""
    return RolloutDecisionContext(
        current_node=cast("Any", object()),
        legal_actions=legal_actions,
        opened_actions=opened_actions,
        openable_actions=openable_actions,
        rollout_step_index=0,
    )


def test_first_openable_action_selector_preserves_openable_order() -> None:
    """First-openable selector returns the first openable action as provided."""
    selector: FirstOpenableActionSelector[Any] = FirstOpenableActionSelector()

    assert selector.choose_action(_context()) == 2


def test_first_openable_action_selector_ignores_opened_actions() -> None:
    """First-openable selector keeps its frontier-only behavior."""
    selector: FirstOpenableActionSelector[Any] = FirstOpenableActionSelector()
    context = _context(
        legal_actions=(0, 1, 2),
        opened_actions=(0,),
        openable_actions=(1, 2),
    )

    assert selector.choose_action(context) == 1


def test_no_rollout_action_selector_stops() -> None:
    """No-rollout selector returns no action."""
    selector: NoRolloutActionSelector[Any] = NoRolloutActionSelector()

    assert selector.choose_action(_context()) is None


def test_random_openable_action_selector_is_reproducible() -> None:
    """Random-openable selector uses its injected random generator."""
    selector_a: RandomOpenableActionSelector[Any] = RandomOpenableActionSelector(
        Random(123)
    )
    selector_b: RandomOpenableActionSelector[Any] = RandomOpenableActionSelector(
        Random(123)
    )
    context = _context(
        legal_actions=(0, 1, 2),
        opened_actions=(0,),
        openable_actions=(1, 2),
    )

    choices_a = [selector_a.choose_action(context) for _ in range(10)]
    choices_b = [selector_b.choose_action(context) for _ in range(10)]

    assert choices_a == choices_b
    assert all(choice in context.openable_actions for choice in choices_a)
    assert 0 not in choices_a


def test_random_openable_action_selector_stops_when_no_action_is_openable() -> None:
    """Random-openable selector returns no action for an empty action set."""
    selector: RandomOpenableActionSelector[Any] = RandomOpenableActionSelector(
        Random(0)
    )

    assert (
        selector.choose_action(_context(legal_actions=(), openable_actions=())) is None
    )
