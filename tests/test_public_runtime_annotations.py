"""Preserve runtime annotation consumers across native deferred annotation changes."""

from typing import get_type_hints

import pytest

from anemone import factory
from anemone.node_evaluation.tree import factory as evaluation_factory
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.rollouts.action_selector import (
    RandomLegalPreferOpenableActionSelector,
    RandomOpenableActionSelector,
)


@pytest.mark.parametrize(
    "target",
    [
        factory.SearchArgs,
        factory.create_search,
        factory.create_search_with_tree_eval_factory,
        factory.create_tree_and_value_branch_selector,
        factory.create_tree_and_value_branch_selector_with_tree_eval_factory,
        factory.create_tree_and_value_exploration,
        factory.create_tree_and_value_exploration_with_tree_eval_factory,
        evaluation_factory.NodeTreeEvaluationFactory.create,
        evaluation_factory.TreeEvaluationConstructor.__call__,
        evaluation_factory.ConfiguredNodeTreeEvaluationFactory.__init__,
        evaluation_factory.ConfiguredNodeTreeEvaluationFactory.create,
        NodeMaxEvaluationFactory.__init__,
        RandomOpenableActionSelector,
        RandomLegalPreferOpenableActionSelector,
    ],
)
def test_public_annotations_remain_resolvable(target: object) -> None:
    """Resolve public factory, protocol and configuration annotations at runtime."""
    assert get_type_hints(target)
