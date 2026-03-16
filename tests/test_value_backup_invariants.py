"""Tests for value backup invariants in ``NodeMinmaxEvaluation``."""

from types import SimpleNamespace
from typing import Any

from valanga import Color
from valanga.evaluations import Certainty

from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from tests.fakes_tree_evaluation import (
    FakeChildEvaluation,
    FakeChildNode,
    set_estimate_value,
)


def _build_parent_eval(
    *,
    turn: Color,
    children: dict[int, FakeChildNode],
    parent_eval_value: float = 0.0,
) -> NodeMinmaxEvaluation[Any, Any]:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=turn),
        branches_children=children,
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)
    set_estimate_value(evaluation, score=parent_eval_value)
    return evaluation


def test_backup_value_equals_a_child_value_and_pv_starts_with_best_branch() -> None:
    children = {
        0: FakeChildNode(
            10,
            FakeChildEvaluation(value_white=0.2, best_branch_sequence=[1]),
        ),
        1: FakeChildNode(
            11,
            FakeChildEvaluation(value_white=0.7, best_branch_sequence=[2]),
        ),
    }
    parent_eval = _build_parent_eval(turn=Color.WHITE, children=children)

    parent_eval.backup_from_children(
        branches_with_updated_value={0, 1}, branches_with_updated_best_branch_seq=set()
    )

    child_values = {c.tree_evaluation.value_white for c in children.values()}
    assert parent_eval.get_score() in child_values
    assert parent_eval.get_score() == 0.7
    assert parent_eval.best_branch_sequence[:1] == [1]


def test_backup_respects_turn_white_max_black_min() -> None:
    children = {
        0: FakeChildNode(10, FakeChildEvaluation(value_white=0.1)),
        1: FakeChildNode(11, FakeChildEvaluation(value_white=0.9)),
    }

    white_parent_eval = _build_parent_eval(turn=Color.WHITE, children=children)
    white_parent_eval.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert white_parent_eval.get_score() == 0.9
    assert white_parent_eval.best_branch_sequence[:1] == [1]

    black_parent_eval = _build_parent_eval(turn=Color.BLACK, children=children)
    black_parent_eval.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )
    assert black_parent_eval.get_score() == 0.1
    assert black_parent_eval.best_branch_sequence[:1] == [0]


def test_backup_tie_break_is_deterministic() -> None:
    children = {
        1: FakeChildNode(20, FakeChildEvaluation(value_white=0.1)),
        0: FakeChildNode(10, FakeChildEvaluation(value_white=0.1)),
    }
    parent_eval = _build_parent_eval(turn=Color.WHITE, children=children)

    parent_eval.backup_from_children(
        branches_with_updated_value={0, 1}, branches_with_updated_best_branch_seq=set()
    )
    first_choice = parent_eval.best_branch_sequence[:1]

    parent_eval.backup_from_children(
        branches_with_updated_value={0, 1}, branches_with_updated_best_branch_seq=set()
    )
    second_choice = parent_eval.best_branch_sequence[:1]

    assert first_choice == second_choice == [0]


def test_setting_estimate_value_sets_canonical_value_views() -> None:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)

    set_estimate_value(evaluation, score=0.33)

    assert evaluation.direct_value is not None
    assert evaluation.minmax_value is not None
    assert evaluation.get_score() == 0.33


def test_setting_estimate_value_populates_direct_value_guardrail() -> None:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)

    set_estimate_value(evaluation, score=0.123)

    assert evaluation.direct_value is not None
    assert evaluation.direct_value.score == 0.123
    assert evaluation.direct_value.over_event is None
    assert evaluation.direct_value.certainty is Certainty.ESTIMATE


def test_setting_estimate_value_keeps_leaf_minmax_in_sync_with_direct_value() -> None:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)

    set_estimate_value(evaluation, score=0.2)
    set_estimate_value(evaluation, score=0.95)

    assert evaluation.direct_value is not None
    assert evaluation.minmax_value is not None
    assert evaluation.direct_value.score == 0.95
    assert evaluation.minmax_value.score == 0.95
