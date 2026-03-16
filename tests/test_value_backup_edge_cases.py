"""Edge-case tests documenting current value backup semantics."""

from types import SimpleNamespace
from typing import Any, cast

from valanga import Color

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
    all_generated: bool,
    parent_eval_value: float,
) -> NodeMinmaxEvaluation[Any, Any]:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=turn),
        branches_children=children,
        all_branches_generated=all_generated,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=cast("Any", parent_tree_node))
    set_estimate_value(evaluation, score=parent_eval_value)
    return evaluation


def test_no_children_values_keeps_direct_value_and_empty_pv() -> None:
    parent_eval = _build_parent_eval(
        turn=Color.WHITE,
        children={},
        all_generated=True,
        parent_eval_value=0.123,
    )

    assert parent_eval.get_score() == 0.123
    assert parent_eval.best_branch() is None
    assert parent_eval.best_branch_sequence == []


# The current implementation in partial mode requires an already-known best branch and
# compares children against the node's direct evaluator value.
def test_partial_expansion_does_not_switch_to_child_worse_than_direct() -> None:
    children = {
        0: FakeChildNode(10, FakeChildEvaluation(value_white=0.2)),
        1: FakeChildNode(11, FakeChildEvaluation(value_white=0.1)),
    }
    parent_eval = _build_parent_eval(
        turn=Color.WHITE,
        children=children,
        all_generated=True,
        parent_eval_value=0.5,
    )

    parent_eval.backup_from_children(
        branches_with_updated_value={0, 1}, branches_with_updated_best_branch_seq=set()
    )
    assert parent_eval.best_branch() is not None

    parent_eval.tree_node.all_branches_generated = False

    # Update a child to a value still worse than direct evaluator (0.4 < 0.5).
    children[1].tree_evaluation.set_value(0.4)
    parent_eval.backup_from_children(
        branches_with_updated_value={1}, branches_with_updated_best_branch_seq=set()
    )

    # In partial mode, if no child beats the direct evaluation, the node keeps its direct value.
    # The implementation can still keep the best explored branch metadata.
    assert parent_eval.get_score() == 0.5
    assert parent_eval.best_branch() == 1
    assert parent_eval.get_score() != 0.4


def test_partial_expansion_switches_to_child_better_than_direct() -> None:
    children = {
        0: FakeChildNode(10, FakeChildEvaluation(value_white=0.2)),
        1: FakeChildNode(11, FakeChildEvaluation(value_white=0.1)),
    }
    parent_eval = _build_parent_eval(
        turn=Color.WHITE,
        children=children,
        all_generated=True,
        parent_eval_value=0.1,
    )

    parent_eval.backup_from_children(
        branches_with_updated_value={0, 1}, branches_with_updated_best_branch_seq=set()
    )
    assert parent_eval.best_branch_sequence[:1] == [0]

    parent_eval.tree_node.all_branches_generated = False

    children[1].tree_evaluation.set_value(0.8)
    parent_eval.backup_from_children(
        branches_with_updated_value={1}, branches_with_updated_best_branch_seq=set()
    )

    assert parent_eval.get_score() == 0.8
    assert parent_eval.best_branch_sequence[:1] == [1]
