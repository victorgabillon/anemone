"""Tests around over-event selection in NodeMinmaxEvaluation."""

from types import SimpleNamespace
from typing import Any

from valanga import Color

from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from tests.fakes_tree_evaluation import (
    FakeChildEvaluation,
    FakeChildNode,
    FakeOverEvent,
)


def _build_parent_eval(
    children: dict[str, FakeChildNode],
) -> NodeMinmaxEvaluation[Any, Any]:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children=children,
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)
    evaluation.over_event = FakeOverEvent()
    return evaluation


def test_becoming_over_prefers_terminal_win_even_if_best_branch_not_over() -> None:
    win_child = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.1,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.WHITE,
                termination="mate",
                _is_over=True,
            ),
        ),
    )
    non_terminal_better_value = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.9,
            over_event=FakeOverEvent(),
        ),
    )

    parent_eval = _build_parent_eval(
        {
            "not_over": non_terminal_better_value,
            "winning_over": win_child,
        }
    )
    # Keep value-ordering with non-terminal branch first.
    parent_eval.branches_sorted_by_value_ = {
        "not_over": (10.0, 0, 2),
        "winning_over": (1.0, 0, 1),
    }

    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_winner(Color.WHITE)
    assert parent_eval.over_event.termination == "mate"


def test_becoming_over_prefers_draw_over_loss_when_all_children_over() -> None:
    draw_child = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.0,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=None,
                termination="stalemate",
                _is_over=True,
            ),
        ),
    )
    loss_child = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=-1.0,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.BLACK,
                termination="mate",
                _is_over=True,
            ),
        ),
    )

    parent_eval = _build_parent_eval({"loss": loss_child, "draw": draw_child})
    parent_eval.branches_sorted_by_value_ = {
        "loss": (5.0, 0, 2),
        "draw": (1.0, 0, 1),
    }

    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_draw()
    assert parent_eval.over_event.termination == "stalemate"


def test_becoming_over_prefers_win_when_multiple_terminal_wins() -> None:
    early_win_child = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.3,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.WHITE,
                termination="mate_in_5",
                _is_over=True,
            ),
        ),
    )
    late_win_child = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.2,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.WHITE,
                termination="mate_in_7",
                _is_over=True,
            ),
        ),
    )

    parent_eval = _build_parent_eval(
        {"first_win": early_win_child, "second_win": late_win_child}
    )
    parent_eval.branches_sorted_by_value_ = {
        "first_win": (1.0, 0, 1),
        "second_win": (2.0, 0, 2),
    }

    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_winner(Color.WHITE)
    assert parent_eval.over_event.termination == "mate_in_5"


def test_becoming_over_uses_terminal_loss_when_no_draw_or_win() -> None:
    first_loss = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=-0.8,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.BLACK,
                termination="mate_a",
                _is_over=True,
            ),
        ),
    )
    second_loss = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=-0.7,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.BLACK,
                termination="mate_b",
                _is_over=True,
            ),
        ),
    )

    # Intentionally leave branches_sorted_by_value_ empty so fallback order is used.
    parent_eval = _build_parent_eval({"a_loss": first_loss, "b_loss": second_loss})
    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_winner(Color.BLACK)
    # Fallback should be deterministic by branch key (`a_loss` before `b_loss`).
    assert parent_eval.over_event.termination == "mate_a"
