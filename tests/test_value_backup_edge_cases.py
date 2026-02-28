"""Edge-case tests documenting current value backup semantics."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from valanga import Color

from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)


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
    best_branch_sequence: list[int] = field(default_factory=list)
    over_event: _FakeOverEvent = field(default_factory=_FakeOverEvent)

    def get_value_white(self) -> float:
        return self.value_white


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
    prev_pv = parent_eval.best_branch_sequence.copy()

    parent_eval.tree_node.all_branches_generated = False

    children[1].tree_evaluation.value_white = 0.4
    parent_eval.minmax_value_update_from_children(branches_with_updated_value={1})

    assert parent_eval.get_value_white() == 0.5
    assert parent_eval.best_branch_sequence == prev_pv
    assert parent_eval.best_branch_sequence[:1] != [1]


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

    children[1].tree_evaluation.value_white = 0.8
    parent_eval.minmax_value_update_from_children(branches_with_updated_value={1})

    assert parent_eval.get_value_white() == 0.8
    assert parent_eval.best_branch_sequence[:1] == [1]
