"""Tests for configurable opening expansion factories."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from anemone.node_evaluation.direct import EvaluationQueries
from anemone.rollouts import (
    FirstOpenableActionSelector,
    NoRolloutActionSelector,
    RandomOpenableActionSelector,
    RolloutOpeningExpansionExecutor,
)
from anemone.tree_manager import (
    AlgorithmNodeTreeManager,
    OnePlyOpeningExpansionExecutor,
    OpeningExpansionConfig,
    OpeningExpansionKind,
    RolloutActionSelectorKind,
    RolloutExpansionConfig,
    TreeExpansions,
    create_opening_expansion_executor,
    create_rollout_action_selector,
)


@dataclass(slots=True)
class _FakeExecutor:
    expanded: bool = False

    def expand(
        self,
        *,
        tree: object,
        opening_instructions: object,
    ) -> TreeExpansions[Any]:
        """Record that the fake executor was used."""
        del tree, opening_instructions
        self.expanded = True
        return TreeExpansions()


def _tree_manager_stub() -> Any:
    """Return the minimal tree-manager surface needed for construction tests."""
    return SimpleNamespace(dynamics=SimpleNamespace())


def test_create_rollout_action_selector_first_openable() -> None:
    """Factory creates the first-openable action selector."""
    selector = create_rollout_action_selector(
        RolloutExpansionConfig(
            action_selector_kind=RolloutActionSelectorKind.FIRST_OPENABLE
        )
    )

    assert isinstance(selector, FirstOpenableActionSelector)


def test_create_rollout_action_selector_no_rollout() -> None:
    """Factory creates the no-rollout action selector."""
    selector = create_rollout_action_selector(
        RolloutExpansionConfig(
            action_selector_kind=RolloutActionSelectorKind.NO_ROLLOUT
        )
    )

    assert isinstance(selector, NoRolloutActionSelector)


def test_create_rollout_action_selector_random_openable() -> None:
    """Factory creates a random-openable selector with a local RNG."""
    selector = create_rollout_action_selector(
        RolloutExpansionConfig(
            action_selector_kind=RolloutActionSelectorKind.RANDOM_OPENABLE,
            random_seed=123,
        )
    )

    assert isinstance(selector, RandomOpenableActionSelector)


def test_rollout_expansion_config_rejects_negative_max_extra_steps() -> None:
    """Rollout expansion config validates the rollout depth limit."""
    with pytest.raises(ValueError, match="max_extra_steps"):
        RolloutExpansionConfig(max_extra_steps=-1)


def test_create_opening_expansion_executor_default_creates_one_ply() -> None:
    """Default opening expansion config preserves one-ply behavior."""
    executor = create_opening_expansion_executor(
        config=OpeningExpansionConfig(),
        tree_manager=cast("Any", _tree_manager_stub()),
        dynamics=cast("Any", object()),
    )

    assert isinstance(executor, OnePlyOpeningExpansionExecutor)


def test_create_opening_expansion_executor_rollout_creates_rollout_executor() -> None:
    """Rollout opening expansion config creates a rollout executor."""
    executor = create_opening_expansion_executor(
        config=OpeningExpansionConfig(
            kind=OpeningExpansionKind.ROLLOUT,
            rollout=RolloutExpansionConfig(max_extra_steps=2),
        ),
        tree_manager=cast("Any", _tree_manager_stub()),
        dynamics=cast("Any", object()),
    )

    assert isinstance(executor, RolloutOpeningExpansionExecutor)
    assert executor.max_extra_steps == 2
    assert executor.stop_on_existing_node


def test_algorithm_node_tree_manager_default_uses_one_ply_executor() -> None:
    """Algorithm manager defaults to one-ply opening expansion."""
    manager = AlgorithmNodeTreeManager(
        tree_manager=cast("Any", _tree_manager_stub()),
        evaluation_queries=EvaluationQueries(),
        node_evaluator=None,
        index_manager=cast("Any", object()),
    )

    assert isinstance(
        manager.opening_expansion_executor, OnePlyOpeningExpansionExecutor
    )


def test_algorithm_node_tree_manager_rollout_config_creates_rollout_executor() -> None:
    """Algorithm manager builds rollout expansion when explicitly configured."""
    manager = AlgorithmNodeTreeManager(
        tree_manager=cast("Any", _tree_manager_stub()),
        evaluation_queries=EvaluationQueries(),
        node_evaluator=None,
        index_manager=cast("Any", object()),
        opening_expansion_config=OpeningExpansionConfig(
            kind=OpeningExpansionKind.ROLLOUT,
            rollout=RolloutExpansionConfig(max_extra_steps=1),
        ),
    )

    assert isinstance(
        manager.opening_expansion_executor, RolloutOpeningExpansionExecutor
    )
    assert manager.opening_expansion_executor.max_extra_steps == 1


def test_algorithm_node_tree_manager_manual_executor_overrides_config() -> None:
    """Explicit executor injection overrides opening expansion config."""
    fake_executor = _FakeExecutor()
    manager = AlgorithmNodeTreeManager(
        tree_manager=cast("Any", _tree_manager_stub()),
        evaluation_queries=EvaluationQueries(),
        node_evaluator=None,
        index_manager=cast("Any", object()),
        opening_expansion_config=OpeningExpansionConfig(
            kind=OpeningExpansionKind.ROLLOUT,
            rollout=RolloutExpansionConfig(max_extra_steps=1),
        ),
        opening_expansion_executor=cast("Any", fake_executor),
    )

    assert manager.opening_expansion_executor is fake_executor
