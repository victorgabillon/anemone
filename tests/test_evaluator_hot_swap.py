"""Focused tests for evaluator hot-swap and provenance on search runtimes."""

from random import Random

import pytest
from valanga import Color

from anemone import SearchArgs, create_search
from anemone.node_evaluation.direct.node_direct_evaluator import DISCOUNT
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
from tests.fake_yaml_game import (
    FakeYamlDynamics,
    FakeYamlState,
    MasterStateValueEvaluatorFromYaml,
)


class _ConcreteFakeYamlState(FakeYamlState):
    """Concrete test state satisfying the abstract ``State`` surface."""

    def pprint(self) -> str:
        """Return a small human-readable representation for test diagnostics."""
        return str(self.node_id)


_CHILDREN_BY_ID = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [],
    4: [],
}

_VALUES_A = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 4.0,
}

_VALUES_B = {
    0: 10.0,
    1: 11.0,
    2: 12.0,
    3: 13.0,
    4: 14.0,
}

_VALUES_C = {
    0: 20.0,
    1: 21.0,
    2: 22.0,
    3: 23.0,
    4: 24.0,
}


def _build_args() -> SearchArgs:
    return SearchArgs(
        node_selector=ComposedNodeSelectorArgs(
            type=NodeSelectorType.COMPOSED,
            priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
            base=UniformArgs(type=NodeSelectorType.UNIFORM),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        stopping_criterion=TreeBranchLimitArgs(
            type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
            tree_branch_limit=10,
        ),
        recommender_rule=SoftmaxRule(type="softmax", temperature=1.0),
        index_computation=None,
    )


def _build_runtime(
    value_by_id: dict[int, float],
):
    starting_state = _ConcreteFakeYamlState(
        node_id=0,
        children_by_id=_CHILDREN_BY_ID,
        turn=Color.WHITE,
    )
    return create_search(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        starting_state=starting_state,
        args=_build_args(),
        random_generator=Random(0),
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(value_by_id),
        state_representation_factory=None,
    )


def _node_at(runtime, *, depth: int, node_id: int):
    return runtime.tree.descendants.node_at(tree_depth=depth, state_tag=node_id)


def _adjusted_score(raw_score: float, *, depth: int) -> float:
    return ((1 / DISCOUNT) ** depth) * raw_score


def test_evaluator_version_starts_at_zero() -> None:
    runtime = _build_runtime(_VALUES_A)

    assert runtime.evaluator_version == 0
    assert runtime.tree.root_node.tree_evaluation.direct_evaluation_version == 0


def test_set_evaluator_increments_version() -> None:
    runtime = _build_runtime(_VALUES_A)

    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(_VALUES_B))
    assert runtime.evaluator_version == 1

    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(_VALUES_C))
    assert runtime.evaluator_version == 2


def test_future_expansions_use_the_new_evaluator() -> None:
    runtime = _build_runtime(_VALUES_A)

    runtime.step()
    child_one = _node_at(runtime, depth=1, node_id=1)
    child_two = _node_at(runtime, depth=1, node_id=2)

    assert child_one.tree_evaluation.direct_value is not None
    assert child_two.tree_evaluation.direct_value is not None
    assert child_one.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(_VALUES_A[1], depth=1)
    )
    assert child_two.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(_VALUES_A[2], depth=1)
    )

    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(_VALUES_B))
    runtime.step()

    grandchild_three = _node_at(runtime, depth=2, node_id=3)
    grandchild_four = _node_at(runtime, depth=2, node_id=4)
    assert grandchild_three.tree_evaluation.direct_value is not None
    assert grandchild_four.tree_evaluation.direct_value is not None
    assert grandchild_three.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(_VALUES_B[3], depth=2)
    )
    assert grandchild_four.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(_VALUES_B[4], depth=2)
    )

    assert child_one.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(_VALUES_A[1], depth=1)
    )
    assert child_two.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(_VALUES_A[2], depth=1)
    )


def test_newly_evaluated_nodes_are_stamped_with_the_current_version() -> None:
    runtime = _build_runtime(_VALUES_A)

    runtime.step()
    child_one = _node_at(runtime, depth=1, node_id=1)
    child_two = _node_at(runtime, depth=1, node_id=2)
    assert child_one.tree_evaluation.direct_evaluation_version == 0
    assert child_two.tree_evaluation.direct_evaluation_version == 0

    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(_VALUES_B))
    runtime.step()

    grandchild_three = _node_at(runtime, depth=2, node_id=3)
    grandchild_four = _node_at(runtime, depth=2, node_id=4)
    assert grandchild_three.tree_evaluation.direct_evaluation_version == 1
    assert grandchild_four.tree_evaluation.direct_evaluation_version == 1


def test_set_evaluator_alone_does_not_modify_existing_nodes() -> None:
    runtime = _build_runtime(_VALUES_A)

    runtime.step()
    child_one = _node_at(runtime, depth=1, node_id=1)
    child_two = _node_at(runtime, depth=1, node_id=2)
    child_one_direct_value = child_one.tree_evaluation.direct_value
    child_two_direct_value = child_two.tree_evaluation.direct_value
    child_one_version = child_one.tree_evaluation.direct_evaluation_version
    child_two_version = child_two.tree_evaluation.direct_evaluation_version

    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(_VALUES_B))

    assert child_one.tree_evaluation.direct_value == child_one_direct_value
    assert child_two.tree_evaluation.direct_value == child_two_direct_value
    assert child_one.tree_evaluation.direct_evaluation_version == child_one_version
    assert child_two.tree_evaluation.direct_evaluation_version == child_two_version
