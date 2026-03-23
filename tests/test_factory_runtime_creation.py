"""Smoke tests for the preferred top-level runtime construction path."""

from random import Random

from valanga import Color

from anemone import (
    SearchArgs,
    SearchRecommender,
    SearchRuntime,
    TreeAndValuePlayerArgs,
    create_search,
    create_tree_and_value_branch_selector,
    create_tree_and_value_exploration,
)
from anemone.factory import (
    SearchArgs as FactorySearchArgs,
)
from anemone.factory import (
    SearchRecommender as FactorySearchRecommender,
)
from anemone.factory import (
    SearchRuntime as FactorySearchRuntime,
)
from anemone.factory import (
    create_search as factory_create_search,
)
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from anemone.recommender_rule.recommender_rule import SoftmaxRule
from anemone.tree_exploration import TreeExploration
from tests.fake_yaml_game import (
    FakeYamlDynamics,
    FakeYamlState,
    MasterStateValueEvaluatorFromYaml,
)


class _ConcreteFakeYamlState(FakeYamlState):
    """Concrete test state satisfying the abstract ``State`` surface."""

    def pprint(self) -> str:
        """Return a small human-readable representation for pyright."""
        return str(self.node_id)


def _build_args() -> SearchArgs:
    """Return a minimal production-style configuration for one fake search."""
    return SearchArgs(
        node_selector=ComposedNodeSelectorArgs(
            type=NodeSelectorType.COMPOSED,
            priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
            base=UniformArgs(type=NodeSelectorType.UNIFORM),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        stopping_criterion=TreeBranchLimitArgs(
            type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
            tree_branch_limit=2,
        ),
        recommender_rule=SoftmaxRule(type="softmax", temperature=1.0),
        index_computation=None,
    )


def test_create_tree_and_value_exploration_returns_runnable_runtime() -> None:
    """The preferred top-level builder should return a ready-to-run runtime."""
    children_by_id = {
        0: [1, 2],
        1: [],
        2: [],
    }
    value_by_id = {
        0: 0.0,
        1: 1.0,
        2: 2.0,
    }
    starting_state = _ConcreteFakeYamlState(
        node_id=0,
        children_by_id=children_by_id,
        turn=Color.WHITE,
    )

    exploration = create_tree_and_value_exploration(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        starting_state=starting_state,
        args=_build_args(),
        random_generator=Random(0),
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(value_by_id),
        state_representation_factory=None,
    )

    assert isinstance(exploration, TreeExploration)
    assert exploration.tree.root_node.state.tag == starting_state.tag

    result = exploration.explore(random_generator=Random(0))

    assert result.tree is exploration.tree
    assert exploration.tree.nodes_count == 3
    assert result.branch_recommendation.branch_evals is not None
    assert (
        result.branch_recommendation.recommended_name
        in result.branch_recommendation.branch_evals
    )


def test_top_level_search_aliases_point_to_runtime_and_wrapper() -> None:
    """The compressed public API should preserve the same concrete objects."""
    children_by_id = {
        0: [1],
        1: [],
    }
    value_by_id = {
        0: 0.0,
        1: 1.0,
    }
    starting_state = _ConcreteFakeYamlState(
        node_id=0,
        children_by_id=children_by_id,
        turn=Color.WHITE,
    )
    args = _build_args()

    runtime = create_search(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        starting_state=starting_state,
        args=args,
        random_generator=Random(0),
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(value_by_id),
        state_representation_factory=None,
    )
    recommender = create_tree_and_value_branch_selector(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        args=args,
        random_generator=Random(0),
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(value_by_id),
        state_representation_factory=None,
    )

    assert SearchArgs is TreeAndValuePlayerArgs
    assert SearchRuntime is TreeExploration
    assert FactorySearchArgs is SearchArgs
    assert FactorySearchRuntime is SearchRuntime
    assert FactorySearchRecommender is SearchRecommender
    assert isinstance(runtime, SearchRuntime)
    assert isinstance(recommender, SearchRecommender)

    factory_runtime = factory_create_search(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        starting_state=starting_state,
        args=args,
        random_generator=Random(0),
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(value_by_id),
        state_representation_factory=None,
    )

    assert isinstance(factory_runtime, FactorySearchRuntime)
