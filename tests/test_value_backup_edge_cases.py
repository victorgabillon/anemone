"""Edge-case tests documenting current value backup semantics."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

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
        return SimpleNamespace(id=self.node_id, state=SimpleNamespace(turn=Color.WHITE))


def _build_parent_eval(
    *,
    turn: Color,
    children: dict[int, _FakeChildNode],
    all_generated: bool,
    parent_eval_value: float,
) -> NodeMinmaxEvaluation[Any, Any]:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=turn),
        branches_children=children,
        all_branches_generated=all_generated,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)
    evaluation.set_evaluation(parent_eval_value)
    return evaluation


def test_no_children_values_keeps_direct_value_and_empty_pv() -> None:
    parent_eval = _build_parent_eval(
        turn=Color.WHITE,
        children={},
        all_generated=True,
        parent_eval_value=0.123,
    )

    assert parent_eval.get_value_white() == 0.123
    assert parent_eval.best_branch() is None
    assert parent_eval.best_branch_sequence == []


# The current implementation in partial mode requires an already-known best branch and
# compares children against the node's direct evaluator value.
def test_partial_expansion_does_not_switch_to_child_worse_than_direct() -> None:
    children = {
        0: _FakeChildNode(10, _FakeChildEvaluation(value_white=0.2)),
        1: _FakeChildNode(11, _FakeChildEvaluation(value_white=0.1)),
    }
    parent_eval = _build_parent_eval(
        turn=Color.WHITE,
        children=children,
        all_generated=True,
        parent_eval_value=0.5,
    )

    parent_eval.minmax_value_update_from_children(branches_with_updated_value={0, 1})
    assert parent_eval.best_branch() is not None

    parent_eval.tree_node.all_branches_generated = False

    # Update a child to a value still worse than direct evaluator (0.4 < 0.5).
    children[1].tree_evaluation.set_value(0.4)
    parent_eval.minmax_value_update_from_children(branches_with_updated_value={1})

    # In partial mode, if no child beats the direct evaluation, the node keeps its direct value.
    # The implementation can still keep the best explored branch metadata.
    assert parent_eval.get_value_white() == 0.5
    assert parent_eval.best_branch() == 1
    assert parent_eval.get_value_white() != 0.4


def test_partial_expansion_switches_to_child_better_than_direct() -> None:
    children = {
        0: _FakeChildNode(10, _FakeChildEvaluation(value_white=0.2)),
        1: _FakeChildNode(11, _FakeChildEvaluation(value_white=0.1)),
    }
    parent_eval = _build_parent_eval(
        turn=Color.WHITE,
        children=children,
        all_generated=True,
        parent_eval_value=0.1,
    )

    parent_eval.minmax_value_update_from_children(branches_with_updated_value={0, 1})
    assert parent_eval.best_branch_sequence[:1] == [0]

    parent_eval.tree_node.all_branches_generated = False

    children[1].tree_evaluation.set_value(0.8)
    parent_eval.minmax_value_update_from_children(branches_with_updated_value={1})

    assert parent_eval.get_value_white() == 0.8
    assert parent_eval.best_branch_sequence[:1] == [1]
