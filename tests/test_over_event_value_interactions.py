"""Tests for over-event and value interaction behaviors in NodeMinmaxEvaluation."""

from valanga import Color

from tests.fakes_tree_evaluation import (
    FakeChildEvaluation,
    FakeChildNode,
    FakeOverEvent,
)
from tests.test_node_minmax_over_event_selection import _build_parent_eval


def test_terminal_win_dominates_terminal_draw_even_if_heuristic_is_lower() -> None:
    win_child = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=-0.9,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.WHITE,
                termination="mate",
                _is_over=True,
            ),
        ),
    )
    high_draw_terminal = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.99,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=None,
                termination="stalemate",
                _is_over=True,
            ),
        ),
    )

    parent_eval = _build_parent_eval({"win": win_child, "draw": high_draw_terminal})

    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_winner(Color.WHITE)
    assert parent_eval.over_event.termination == "mate"


def test_terminal_draw_dominates_terminal_loss() -> None:
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
    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_draw()
    assert parent_eval.over_event.termination == "stalemate"


def test_all_terminal_children_choose_best_terminal_outcome() -> None:
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
    win_child = FakeChildNode(
        node_id=2,
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

    parent_eval = _build_parent_eval({"draw": draw_child, "win": win_child})

    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_winner(Color.WHITE)
    assert parent_eval.over_event.termination == "mate"


def test_turn_changes_terminal_preference_between_winning_children() -> None:
    white_win = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.5,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.WHITE,
                termination="white_mate",
                _is_over=True,
            ),
        ),
    )
    black_win = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=-0.5,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.BLACK,
                termination="black_mate",
                _is_over=True,
            ),
        ),
    )

    white_parent = _build_parent_eval({"w": white_win, "b": black_win})
    white_parent.becoming_over_from_children()

    black_parent = _build_parent_eval({"w": white_win, "b": black_win})
    black_parent.tree_node.state.turn = Color.BLACK
    black_parent.becoming_over_from_children()

    assert white_parent.over_event.termination == "white_mate"
    assert black_parent.over_event.termination == "black_mate"


def test_terminal_unknown_winner_is_chosen_over_loss() -> None:
    unknown_terminal = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.1,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=None,
                termination="unknown_terminal",
                _is_over=True,
            ),
        ),
    )
    loss_child = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=-0.7,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.BLACK,
                termination="mate",
                _is_over=True,
            ),
        ),
    )

    parent_eval = _build_parent_eval({"u": unknown_terminal, "l": loss_child})

    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.termination == "unknown_terminal"


def test_terminal_selection_and_minmax_best_branch_remain_consistent() -> None:
    win_child = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=-0.9,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.WHITE,
                termination="mate",
                _is_over=True,
            ),
        ),
    )
    draw_child = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.2,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=None,
                termination="stalemate",
                _is_over=True,
            ),
        ),
    )

    parent_eval = _build_parent_eval({"draw": draw_child, "win": win_child})
    parent_eval.tree_node.all_branches_generated = True
    parent_eval.set_evaluation(0.0)

    parent_eval.becoming_over_from_children()
    parent_eval.minmax_value_update_from_children({"draw", "win"})

    assert parent_eval.over_event.is_winner(Color.WHITE)
    # Now: terminal over-event selection and PV tracking are consistent.
    assert parent_eval.best_branch_sequence[:1] == ["win"]
    assert parent_eval.over_event.termination == "mate"
