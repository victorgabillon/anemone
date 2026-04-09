"""Focused tests for explicit reevaluation of existing tree nodes."""

from collections.abc import Sequence
from random import Random
from typing import Any

import pytest
from valanga import Color, State
from valanga.evaluations import EvalItem, Value

from anemone import SearchArgs, create_search
from anemone.indices.node_indices.index_types import IndexComputationType
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


class _RecordingYamlEvaluator(MasterStateValueEvaluatorFromYaml):
    """YAML-backed evaluator that records the node ids sent for batch scoring."""

    def __init__(self, value_by_id: dict[int, float]) -> None:
        super().__init__(value_by_id)
        self.batch_history: list[list[int]] = []

    def evaluate_batch_items[ItemStateT: State](
        self, items: Sequence[EvalItem[ItemStateT]]
    ) -> list[Value]:
        self.batch_history.append(
            [int(getattr(getattr(item, "state", item), "tag")) for item in items]
        )
        return super().evaluate_batch_items(items)


_CHILDREN_BY_ID = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [],
    4: [],
}


def _build_args(
    *,
    index_computation: IndexComputationType | None = None,
) -> SearchArgs:
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
        index_computation=index_computation,
    )


def _build_runtime(
    *,
    value_by_id: dict[int, float],
    index_computation: IndexComputationType | None = None,
    evaluator: MasterStateValueEvaluatorFromYaml | None = None,
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
        args=_build_args(index_computation=index_computation),
        random_generator=Random(0),
        master_state_value_evaluator=evaluator
        or MasterStateValueEvaluatorFromYaml(value_by_id),
        state_representation_factory=None,
    )


def _node_at(runtime: Any, *, depth: int, node_id: int) -> Any:
    return runtime.tree.descendants.node_at(tree_depth=depth, state_tag=node_id)


def _adjusted_score(raw_score: float, *, depth: int) -> float:
    return ((1 / DISCOUNT) ** depth) * raw_score


def test_reevaluate_existing_root_updates_direct_value_and_version() -> None:
    values_a = {
        0: 0.9,
        1: 0.2,
        2: 0.1,
        3: 0.0,
        4: 0.0,
    }
    values_b = {
        0: 0.05,
        1: 0.2,
        2: 0.1,
        3: 0.0,
        4: 0.0,
    }
    runtime = _build_runtime(value_by_id=values_a)

    runtime.step()
    root = runtime.tree.root_node
    assert root.tree_evaluation.direct_value is not None
    assert root.tree_evaluation.direct_value.score == pytest.approx(values_a[0])

    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(values_b))
    report = runtime.reevaluate_nodes([root])

    assert report.evaluator_version == 1
    assert report.requested_count == 1
    assert report.reevaluated_count == 1
    assert report.changed_count == 1
    assert report.skipped_terminal_count == 0
    assert root.tree_evaluation.direct_value is not None
    assert root.tree_evaluation.direct_value.score == pytest.approx(values_b[0])
    assert root.tree_evaluation.direct_evaluation_version == 1


def test_reevaluate_existing_nodes_updates_root_value_and_best_branch() -> None:
    values_a = {
        0: 0.0,
        1: 0.8,
        2: 0.3,
        3: 0.0,
        4: 0.0,
    }
    values_b = {
        0: 0.0,
        1: 0.1,
        2: 0.7,
        3: 0.0,
        4: 0.0,
    }
    runtime = _build_runtime(value_by_id=values_a)

    runtime.step()
    root = runtime.tree.root_node
    child_one = _node_at(runtime, depth=1, node_id=1)
    child_two = _node_at(runtime, depth=1, node_id=2)
    assert root.tree_evaluation.best_branch() == 0
    assert root.tree_evaluation.best_branch_sequence == [0]
    assert root.tree_evaluation.get_value().score == pytest.approx(
        _adjusted_score(values_a[1], depth=1)
    )

    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(values_b))
    report = runtime.reevaluate_nodes([child_one, child_two])

    assert report.requested_count == 2
    assert report.reevaluated_count == 2
    assert report.changed_count == 2
    assert child_one.tree_evaluation.direct_evaluation_version == 1
    assert child_two.tree_evaluation.direct_evaluation_version == 1
    assert child_one.tree_evaluation.direct_value is not None
    assert child_two.tree_evaluation.direct_value is not None
    assert child_one.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(values_b[1], depth=1)
    )
    assert child_two.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(values_b[2], depth=1)
    )
    assert root.tree_evaluation.best_branch() == 1
    assert root.tree_evaluation.best_branch_sequence == [1]
    assert root.tree_evaluation.get_value().score == pytest.approx(
        _adjusted_score(values_b[2], depth=1)
    )


def test_reevaluate_nodes_refreshes_exploration_indices() -> None:
    values_a = {
        0: 0.0,
        1: 0.8,
        2: 0.3,
        3: 0.0,
        4: 0.0,
    }
    values_b = {
        0: 0.0,
        1: 0.1,
        2: 0.7,
        3: 0.0,
        4: 0.0,
    }
    runtime = _build_runtime(
        value_by_id=values_a,
        index_computation=IndexComputationType.RECUR_ZIPF,
    )

    runtime.step()
    child_one = _node_at(runtime, depth=1, node_id=1)
    child_two = _node_at(runtime, depth=1, node_id=2)
    assert child_one.exploration_index_data is not None
    assert child_two.exploration_index_data is not None
    assert child_two.exploration_index_data.index == pytest.approx(-0.25)

    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(values_b))
    runtime.reevaluate_nodes([child_one, child_two])

    assert child_two.exploration_index_data.index == pytest.approx(-0.5)


def test_reevaluate_nodes_deduplicates_requests_before_batch_evaluation() -> None:
    values_a = {
        0: 0.0,
        1: 0.8,
        2: 0.3,
        3: 0.0,
        4: 0.0,
    }
    values_b = {
        0: 0.0,
        1: 0.4,
        2: 0.6,
        3: 0.0,
        4: 0.0,
    }
    runtime = _build_runtime(value_by_id=values_a)

    runtime.step()
    child_one = _node_at(runtime, depth=1, node_id=1)
    child_two = _node_at(runtime, depth=1, node_id=2)
    recording_evaluator = _RecordingYamlEvaluator(values_b)
    runtime.set_evaluator(recording_evaluator)

    report = runtime.reevaluate_nodes([child_one, child_one, child_two, child_two])

    assert report.requested_count == 4
    assert report.reevaluated_count == 2
    assert report.changed_count == 2
    assert recording_evaluator.batch_history == [[1, 2]]


def test_reevaluate_nodes_skips_terminal_states() -> None:
    values_a = {
        0: 0.0,
        1: 0.8,
        2: 0.3,
        3: 0.4,
        4: 0.6,
    }
    values_b = {
        0: 0.0,
        1: 0.1,
        2: 0.7,
        3: 0.9,
        4: 0.2,
    }
    runtime = _build_runtime(value_by_id=values_a)

    runtime.step()
    runtime.step()
    leaf_three = _node_at(runtime, depth=2, node_id=3)
    leaf_four = _node_at(runtime, depth=2, node_id=4)
    leaf_three_value = leaf_three.tree_evaluation.direct_value
    leaf_four_value = leaf_four.tree_evaluation.direct_value
    leaf_three_version = leaf_three.tree_evaluation.direct_evaluation_version
    leaf_four_version = leaf_four.tree_evaluation.direct_evaluation_version

    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(values_b))
    report = runtime.reevaluate_nodes([leaf_three, leaf_four])

    assert report.requested_count == 2
    assert report.reevaluated_count == 0
    assert report.changed_count == 0
    assert report.skipped_terminal_count == 2
    assert leaf_three.tree_evaluation.direct_value == leaf_three_value
    assert leaf_four.tree_evaluation.direct_value == leaf_four_value
    assert leaf_three.tree_evaluation.direct_evaluation_version == leaf_three_version
    assert leaf_four.tree_evaluation.direct_evaluation_version == leaf_four_version


def test_reevaluate_nodes_empty_request_is_a_no_op() -> None:
    values_a = {
        0: 0.0,
        1: 0.8,
        2: 0.3,
        3: 0.0,
        4: 0.0,
    }
    runtime = _build_runtime(value_by_id=values_a)

    runtime.step()
    root_value_before = runtime.tree.root_node.tree_evaluation.get_value()

    report = runtime.reevaluate_nodes([])

    assert report.evaluator_version == 0
    assert report.requested_count == 0
    assert report.reevaluated_count == 0
    assert report.changed_count == 0
    assert report.skipped_terminal_count == 0
    assert runtime.tree.root_node.tree_evaluation.get_value() == root_value_before
