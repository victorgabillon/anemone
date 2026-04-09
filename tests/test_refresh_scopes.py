"""Focused tests for scoped reevaluation helpers and refresh orchestration."""

from random import Random
from typing import Any

import pytest
from valanga import Color

from anemone import SearchArgs, create_search
from anemone.node_evaluation.direct.node_direct_evaluator import DISCOUNT
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.nodes.utils import best_node_sequence_from_node
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


_PV_CHILDREN_BY_ID = {
    0: [1, 2],
    1: [3, 4],
    2: [5, 6],
    3: [7],
    4: [],
    5: [],
    6: [],
    7: [],
}

_PV_VALUES_A = {
    0: 0.0,
    1: 0.8,
    2: 0.4,
    3: 0.9,
    4: 0.1,
    5: 0.2,
    6: 0.0,
    7: 0.7,
}

_PV_VALUES_B = {
    0: 0.05,
    1: 0.6,
    2: 0.3,
    3: 1.1,
    4: 0.15,
    5: 0.25,
    6: 0.05,
    7: 0.75,
}

_FRONTIER_CHILDREN_BY_ID = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [5],
    4: [6],
    5: [],
    6: [],
}

_FRONTIER_VALUES_A = {
    0: 0.0,
    1: 0.9,
    2: 0.4,
    3: 0.7,
    4: 0.6,
    5: 0.2,
    6: 0.1,
}

_FRONTIER_VALUES_B = {
    0: 0.1,
    1: 0.5,
    2: 0.45,
    3: 1.0,
    4: 0.8,
    5: 0.25,
    6: 0.15,
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
            tree_branch_limit=20,
        ),
        recommender_rule=SoftmaxRule(type="softmax", temperature=1.0),
        index_computation=None,
    )


def _build_runtime(
    *,
    children_by_id: dict[int, list[int]],
    value_by_id: dict[int, float],
):
    starting_state = _ConcreteFakeYamlState(
        node_id=0,
        children_by_id=children_by_id,
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


def _node_at(runtime: Any, *, depth: int, node_id: int) -> Any:
    return runtime.tree.descendants.node_at(tree_depth=depth, state_tag=node_id)


def _run_steps(runtime: Any, count: int) -> None:
    for _ in range(count):
        runtime.step()


def _direct_versions_by_node_id(runtime: Any) -> dict[int, int | None]:
    return {
        int(node.tag): node.tree_evaluation.direct_evaluation_version
        for _, _, node in runtime.tree.descendants.iter_on_all_nodes()
    }


def _adjusted_score(raw_score: float, *, depth: int) -> float:
    return ((1 / DISCOUNT) ** depth) * raw_score


def _pv_node_ids(runtime: Any) -> list[int]:
    return [
        int(node.tag)
        for node in best_node_sequence_from_node(runtime.tree.root_node)[1:]
    ]


def test_reevaluate_root_children_selects_only_root_children() -> None:
    runtime = _build_runtime(
        children_by_id=_PV_CHILDREN_BY_ID,
        value_by_id=_PV_VALUES_A,
    )
    _run_steps(runtime, 2)
    versions_before = _direct_versions_by_node_id(runtime)

    report = runtime.refresh_with_evaluator(
        MasterStateValueEvaluatorFromYaml(_PV_VALUES_B),
        scope="root_children",
    )

    assert report.requested_count == 2
    assert report.reevaluated_count == 2
    assert _direct_versions_by_node_id(runtime) == {
        0: versions_before[0],
        1: 1,
        2: 1,
        3: versions_before[3],
        4: versions_before[4],
        5: versions_before[5],
        6: versions_before[6],
    }


def test_reevaluate_pv_selects_only_current_principal_variation() -> None:
    runtime = _build_runtime(
        children_by_id=_PV_CHILDREN_BY_ID,
        value_by_id=_PV_VALUES_A,
    )
    _run_steps(runtime, 2)
    pv_node_ids = _pv_node_ids(runtime)
    versions_before = _direct_versions_by_node_id(runtime)

    report = runtime.refresh_with_evaluator(
        MasterStateValueEvaluatorFromYaml(_PV_VALUES_B),
        scope="pv",
    )

    assert report.requested_count == 2
    assert pv_node_ids == [1, 4]
    assert report.reevaluated_count == 1
    assert report.skipped_terminal_count == 1
    assert runtime.tree.root_node.tree_evaluation.best_branch_sequence == [0, 1]
    versions_after = _direct_versions_by_node_id(runtime)
    assert versions_after[1] == 1
    assert versions_after[4] == versions_before[4]
    assert versions_after[2] == versions_before[2]
    assert versions_after[3] == versions_before[3]
    assert versions_after[5] == versions_before[5]
    assert versions_after[6] == versions_before[6]


def test_reevaluate_frontier_without_limit_selects_all_frontier_nodes() -> None:
    runtime = _build_runtime(
        children_by_id=_FRONTIER_CHILDREN_BY_ID,
        value_by_id=_FRONTIER_VALUES_A,
    )
    _run_steps(runtime, 2)
    versions_before = _direct_versions_by_node_id(runtime)

    report = runtime.reevaluate_frontier()

    assert report.evaluator_version == 0
    assert report.requested_count == 2
    assert report.reevaluated_count == 2
    assert _direct_versions_by_node_id(runtime) == {
        0: versions_before[0],
        1: versions_before[1],
        2: versions_before[2],
        3: 0,
        4: 0,
    }


def test_reevaluate_frontier_respects_k_in_deterministic_order() -> None:
    runtime = _build_runtime(
        children_by_id=_FRONTIER_CHILDREN_BY_ID,
        value_by_id=_FRONTIER_VALUES_A,
    )
    _run_steps(runtime, 2)
    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(_FRONTIER_VALUES_B))
    versions_before = _direct_versions_by_node_id(runtime)

    report = runtime.reevaluate_frontier(k=1)

    assert report.requested_count == 1
    assert report.reevaluated_count == 1
    # Frontier traversal follows the current ordered frontier branches, which
    # makes node 3 the deterministic first frontier node in this tree.
    assert _direct_versions_by_node_id(runtime) == {
        0: versions_before[0],
        1: versions_before[1],
        2: versions_before[2],
        3: 1,
        4: versions_before[4],
    }


def test_reevaluate_frontier_with_zero_limit_is_a_no_op() -> None:
    runtime = _build_runtime(
        children_by_id=_FRONTIER_CHILDREN_BY_ID,
        value_by_id=_FRONTIER_VALUES_A,
    )
    _run_steps(runtime, 2)
    versions_before = _direct_versions_by_node_id(runtime)

    report = runtime.reevaluate_frontier(k=0)

    assert report.requested_count == 0
    assert report.reevaluated_count == 0
    assert report.changed_count == 0
    assert _direct_versions_by_node_id(runtime) == versions_before


def test_reevaluate_frontier_with_negative_limit_raises() -> None:
    runtime = _build_runtime(
        children_by_id=_FRONTIER_CHILDREN_BY_ID,
        value_by_id=_FRONTIER_VALUES_A,
    )
    _run_steps(runtime, 2)

    with pytest.raises(ValueError, match="k must be non-negative or None"):
        runtime.reevaluate_frontier(k=-1)


def test_reevaluate_leaves_selects_current_structural_leaves() -> None:
    runtime = _build_runtime(
        children_by_id=_FRONTIER_CHILDREN_BY_ID,
        value_by_id=_FRONTIER_VALUES_A,
    )
    _run_steps(runtime, 2)
    runtime.set_evaluator(MasterStateValueEvaluatorFromYaml(_FRONTIER_VALUES_B))
    versions_before = _direct_versions_by_node_id(runtime)

    report = runtime.reevaluate_leaves()

    assert report.requested_count == 2
    assert report.reevaluated_count == 2
    assert _direct_versions_by_node_id(runtime) == {
        0: versions_before[0],
        1: versions_before[1],
        2: versions_before[2],
        3: 1,
        4: 1,
    }


def test_refresh_with_evaluator_root_children_swaps_and_reevaluates() -> None:
    runtime = _build_runtime(
        children_by_id=_PV_CHILDREN_BY_ID,
        value_by_id=_PV_VALUES_A,
    )
    _run_steps(runtime, 2)
    child_one = _node_at(runtime, depth=1, node_id=1)
    child_two = _node_at(runtime, depth=1, node_id=2)
    depth_two_before = {
        3: _node_at(
            runtime, depth=2, node_id=3
        ).tree_evaluation.direct_evaluation_version,
        4: _node_at(
            runtime, depth=2, node_id=4
        ).tree_evaluation.direct_evaluation_version,
        5: _node_at(
            runtime, depth=2, node_id=5
        ).tree_evaluation.direct_evaluation_version,
        6: _node_at(
            runtime, depth=2, node_id=6
        ).tree_evaluation.direct_evaluation_version,
    }

    report = runtime.refresh_with_evaluator(
        MasterStateValueEvaluatorFromYaml(_PV_VALUES_B),
        scope="root_children",
    )

    assert report.evaluator_version == 1
    assert runtime.evaluator_version == 1
    assert child_one.tree_evaluation.direct_value is not None
    assert child_two.tree_evaluation.direct_value is not None
    assert child_one.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(_PV_VALUES_B[1], depth=1)
    )
    assert child_two.tree_evaluation.direct_value.score == pytest.approx(
        _adjusted_score(_PV_VALUES_B[2], depth=1)
    )
    assert child_one.tree_evaluation.direct_evaluation_version == 1
    assert child_two.tree_evaluation.direct_evaluation_version == 1
    assert (
        _node_at(runtime, depth=2, node_id=3).tree_evaluation.direct_evaluation_version
        == depth_two_before[3]
    )
    assert (
        _node_at(runtime, depth=2, node_id=4).tree_evaluation.direct_evaluation_version
        == depth_two_before[4]
    )
    assert (
        _node_at(runtime, depth=2, node_id=5).tree_evaluation.direct_evaluation_version
        == depth_two_before[5]
    )
    assert (
        _node_at(runtime, depth=2, node_id=6).tree_evaluation.direct_evaluation_version
        == depth_two_before[6]
    )


def test_refresh_with_evaluator_defaults_to_frontier_scope() -> None:
    runtime_default = _build_runtime(
        children_by_id=_FRONTIER_CHILDREN_BY_ID,
        value_by_id=_FRONTIER_VALUES_A,
    )
    runtime_explicit = _build_runtime(
        children_by_id=_FRONTIER_CHILDREN_BY_ID,
        value_by_id=_FRONTIER_VALUES_A,
    )
    _run_steps(runtime_default, 2)
    _run_steps(runtime_explicit, 2)

    default_report = runtime_default.refresh_with_evaluator(
        MasterStateValueEvaluatorFromYaml(_FRONTIER_VALUES_B)
    )
    runtime_explicit.set_evaluator(
        MasterStateValueEvaluatorFromYaml(_FRONTIER_VALUES_B)
    )
    explicit_report = runtime_explicit.reevaluate_frontier()

    assert default_report == explicit_report
    assert _direct_versions_by_node_id(runtime_default) == _direct_versions_by_node_id(
        runtime_explicit
    )


def test_refresh_with_evaluator_rejects_invalid_scope() -> None:
    runtime = _build_runtime(
        children_by_id=_FRONTIER_CHILDREN_BY_ID,
        value_by_id=_FRONTIER_VALUES_A,
    )

    with pytest.raises(ValueError) as exc_info:
        runtime.refresh_with_evaluator(
            MasterStateValueEvaluatorFromYaml(_FRONTIER_VALUES_B),
            scope="unknown",
        )

    assert runtime.evaluator_version == 0
    assert "Supported scopes are" in str(exc_info.value)
    assert "frontier" in str(exc_info.value)
    assert "pv_and_root_children" in str(exc_info.value)


def test_reevaluate_root_children_is_no_op_when_root_has_no_children() -> None:
    runtime = _build_runtime(children_by_id={0: []}, value_by_id={0: 0.5})

    report = runtime.reevaluate_root_children()

    assert report.requested_count == 0
    assert report.reevaluated_count == 0
    assert report.changed_count == 0
    assert report.skipped_terminal_count == 0


def test_refresh_with_evaluator_pv_and_root_children_deduplicates_union() -> None:
    runtime = _build_runtime(
        children_by_id=_PV_CHILDREN_BY_ID,
        value_by_id=_PV_VALUES_A,
    )
    _run_steps(runtime, 2)
    pv_node_ids = _pv_node_ids(runtime)
    versions_before = _direct_versions_by_node_id(runtime)

    report = runtime.refresh_with_evaluator(
        MasterStateValueEvaluatorFromYaml(_PV_VALUES_B),
        scope="pv_and_root_children",
    )

    assert report.requested_count == 3
    assert pv_node_ids == [1, 4]
    assert report.reevaluated_count == 2
    assert report.skipped_terminal_count == 1
    versions_after = _direct_versions_by_node_id(runtime)
    assert versions_after[1] == 1
    assert versions_after[2] == 1
    assert versions_after[4] == versions_before[4]
    assert versions_after[3] == versions_before[3]
    assert versions_after[5] == versions_before[5]
    assert versions_after[6] == versions_before[6]


def test_refresh_with_evaluator_all_scope_reevaluates_all_current_nonterminal_nodes() -> (
    None
):
    runtime = _build_runtime(
        children_by_id=_PV_CHILDREN_BY_ID,
        value_by_id=_PV_VALUES_A,
    )
    _run_steps(runtime, 2)

    report = runtime.refresh_with_evaluator(
        MasterStateValueEvaluatorFromYaml(_PV_VALUES_B),
        scope="all",
    )

    assert report.requested_count == 7
    assert report.reevaluated_count == 4
    assert report.skipped_terminal_count == 3
    assert _direct_versions_by_node_id(runtime) == {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 0,
        5: 0,
        6: 0,
    }
