"""Access live runtime RNGs without replacing them or consuming random draws."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from anemone.tree_exploration import TreeExploration


def rollout_random_generator(search: TreeExploration[Any]) -> Random | None:
    """Find the RNG on the executor's actual selector, including runtime overrides."""
    executor = search.tree_manager.opening_expansion_executor
    selector = getattr(executor, "rollout_action_selector", None)
    random_generator = getattr(selector, "random_generator", None)
    return random_generator if isinstance(random_generator, Random) else None


def restore_random_state(random_generator: Random, state: object) -> None:
    """Restore typed or JSON state into an existing independent RNG."""
    # JSON converts both the outer state and its internal vector into lists.
    version, state_vector, gaussian = cast(
        "tuple[int, tuple[int, ...], float | None]", state
    )
    random_generator.setstate((version, tuple(state_vector), gaussian))
