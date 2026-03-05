"""Tests for value backup invariants in ``NodeMinmaxEvaluation``."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from valanga import Color

from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.values import Certainty, Value


@dataclass
class _FakeOverEvent:
    def is_over(self) -> bool:
        return False

    def is_winner(self, player: Color) -> bool:
        del player
        return False

    def is_draw(self) -> bool:
        return False


@dataclass
class _FakeChildEvaluation:
    value_white: float
    value_white_minmax: float | None = None
    direct_value: Value | None = None
    minmax_value: Value | None = None
    best_branch_sequence: list[int] = field(default_factory=list)
    over_event: _FakeOverEvent = field(default_factory=_FakeOverEvent)

    def __post_init__(self) -> None:
        canonical_value = Value(
            score=self.value_white,
            certainty=(
                Certainty.FORCED if self.over_event.is_over() else Certainty.ESTIMATE
            ),
            over_event=self.over_event if self.over_event.is_over() else None,
        )
        if self.direct_value is None:
            self.direct_value = canonical_value
        if self.minmax_value is None:
            self.minmax_value = self.direct_value
        self.value_white_minmax = self.minmax_value.score

    def set_value(self, score: float) -> None:
        """Keep float bridge and canonical Values aligned in test mutations."""
        self.value_white = score
        canonical_value = Value(
            score=score,
            certainty=(
                Certainty.FORCED if self.over_event.is_over() else Certainty.ESTIMATE
            ),
            over_event=self.over_event if self.over_event.is_over() else None,
        )
        self.direct_value = canonical_value
        self.minmax_value = canonical_value
        self.value_white_minmax = canonical_value.score

    def get_value_white(self) -> float:
        if self.minmax_value is not None:
            return self.minmax_value.score
        assert self.direct_value is not None
        return self.direct_value.score


@dataclass
class _FakeChildNode:
    node_id: int
    tree_evaluation: _FakeChildEvaluation

    @property
    def tree_node(self) -> Any:
        return SimpleNamespace(id=self.node_id, state=SimpleNamespace(turn=Color.BLACK))


def _build_parent_eval(
    *,
    turn: Color,
    children: dict[int, _FakeChildNode],
    parent_eval_value: float = 0.0,
) -> NodeMinmaxEvaluation[Any, Any]:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=turn),
        branches_children=children,
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)
    evaluation.set_evaluation(parent_eval_value)
    return evaluation


def test_backup_value_equals_a_child_value_and_pv_starts_with_best_branch() -> None:
    children = {
        0: _FakeChildNode(
            10,
            _FakeChildEvaluation(value_white=0.2, best_branch_sequence=[1]),
        ),
        1: _FakeChildNode(
            11,
            _FakeChildEvaluation(value_white=0.7, best_branch_sequence=[2]),
        ),
    }
    parent_eval = _build_parent_eval(turn=Color.WHITE, children=children)

    parent_eval.minmax_value_update_from_children(branches_with_updated_value={0, 1})

    child_values = {c.tree_evaluation.value_white for c in children.values()}
    assert parent_eval.value_white_minmax in child_values
    assert parent_eval.value_white_minmax == 0.7
    assert parent_eval.best_branch_sequence[:1] == [1]


def test_backup_respects_turn_white_max_black_min() -> None:
    children = {
        0: _FakeChildNode(10, _FakeChildEvaluation(value_white=0.1)),
        1: _FakeChildNode(11, _FakeChildEvaluation(value_white=0.9)),
    }

    white_parent_eval = _build_parent_eval(turn=Color.WHITE, children=children)
    white_parent_eval.minmax_value_update_from_children(
        branches_with_updated_value={0, 1}
    )
    assert white_parent_eval.value_white_minmax == 0.9
    assert white_parent_eval.best_branch_sequence[:1] == [1]

    black_parent_eval = _build_parent_eval(turn=Color.BLACK, children=children)
    black_parent_eval.minmax_value_update_from_children(
        branches_with_updated_value={0, 1}
    )
    assert black_parent_eval.get_score() == 0.1
    assert black_parent_eval.best_branch_sequence[:1] == [0]


def test_backup_tie_break_is_deterministic() -> None:
    children = {
        1: _FakeChildNode(20, _FakeChildEvaluation(value_white=0.1)),
        0: _FakeChildNode(10, _FakeChildEvaluation(value_white=0.1)),
    }
    parent_eval = _build_parent_eval(turn=Color.WHITE, children=children)

    parent_eval.minmax_value_update_from_children(branches_with_updated_value={0, 1})
    first_choice = parent_eval.best_branch_sequence[:1]

    parent_eval.minmax_value_update_from_children(branches_with_updated_value={0, 1})
    second_choice = parent_eval.best_branch_sequence[:1]

    assert first_choice == second_choice == [0]


def test_float_bridge_properties_are_read_only_views() -> None:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)

    evaluation.set_evaluation(0.33)

    assert evaluation.value_white_direct_evaluation == 0.33
    assert evaluation.value_white_minmax == 0.33
    with pytest.raises(AttributeError):
        evaluation.value_white_minmax = 0.99


def test_set_evaluation_populates_direct_value_guardrail() -> None:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)

    evaluation.set_evaluation(0.123)

    assert evaluation.direct_value is not None
    assert evaluation.direct_value.score == 0.123
    assert evaluation.direct_value.over_event is None
    assert evaluation.direct_value.certainty is Certainty.ESTIMATE


def test_set_evaluation_keeps_leaf_minmax_in_sync_with_direct_value() -> None:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)

    evaluation.set_evaluation(0.2)
    evaluation.set_evaluation(0.95)

    assert evaluation.direct_value is not None
    assert evaluation.minmax_value is not None
    assert evaluation.direct_value.score == 0.95
    assert evaluation.minmax_value.score == 0.95
