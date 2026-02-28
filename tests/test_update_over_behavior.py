"""Tests for incremental terminal propagation via NodeMinmaxEvaluation.update_over()."""

from typing import Hashable

from valanga import Color

from tests.test_node_minmax_over_event_selection import (
    FakeChildEvaluation,
    FakeChildNode,
    FakeOverEvent,
    _build_parent_eval,
)


def _terminal_child(*, node_id: int, winner: Color | None, term: str) -> FakeChildNode:
    return FakeChildNode(
        node_id=node_id,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.0,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=winner,
                termination=term,
                _is_over=True,
            ),
        ),
    )


def _non_terminal_child(*, node_id: int, value_white: float = 0.0) -> FakeChildNode:
    return FakeChildNode(
        node_id=node_id,
        tree_evaluation=FakeChildEvaluation(
            value_white=value_white,
            over_event=FakeOverEvent(_is_over=False),
        ),
    )


def _update_over(parent_eval, branches: set[Hashable]) -> bool:
    """Return whether parent became newly over (robust to different return signatures)."""
    result = parent_eval.update_over(branches)
    if isinstance(result, tuple):
        # Common patterns: (has_value_changed, is_newly_over) or (is_newly_over, ...)
        # Prefer last bool if tuple of bools; otherwise pick first bool.
        bools = [x for x in result if isinstance(x, bool)]
        if bools:
            return bools[-1]
    assert isinstance(result, bool)
    return result


def test_update_over_single_terminal_win_forces_parent_over() -> None:
    win = _terminal_child(node_id=1, winner=Color.WHITE, term="mate")
    live = _non_terminal_child(node_id=2, value_white=0.2)

    parent = _build_parent_eval({"win": win, "live": live})
    parent.set_evaluation(0.0)
    parent.tree_node.state.turn = Color.WHITE

    newly_over = _update_over(parent, {"win"})

    assert newly_over
    assert parent.over_event.is_over()
    assert parent.over_event.is_winner(Color.WHITE)
    assert parent.over_event.termination == "mate"


def test_update_over_single_terminal_draw_does_not_force_over_if_other_live_children() -> None:
    draw = _terminal_child(node_id=1, winner=None, term="stalemate")
    live = _non_terminal_child(node_id=2, value_white=0.2)

    parent = _build_parent_eval({"draw": draw, "live": live})
    parent.set_evaluation(0.0)
    parent.tree_node.state.turn = Color.WHITE

    newly_over = _update_over(parent, {"draw"})

    # pin current behavior: draw alone is not enough to conclude game is over at parent
    assert not newly_over
    assert not parent.over_event.is_over()


def test_update_over_all_children_terminal_forces_parent_over() -> None:
    draw = _terminal_child(node_id=1, winner=None, term="stalemate")
    loss = _terminal_child(
        node_id=2,
        winner=Color.BLACK,
        term="mate",
    )

    parent = _build_parent_eval({"draw": draw, "loss": loss})
    parent.set_evaluation(0.0)
    parent.tree_node.state.turn = Color.WHITE

    newly_over = _update_over(parent, {"draw", "loss"})

    assert newly_over
    assert parent.over_event.is_over()
    assert parent.over_event.termination == "stalemate"


def test_update_over_single_terminal_loss_does_not_force_over_if_other_live_children() -> None:
    loss = _terminal_child(node_id=1, winner=Color.BLACK, term="mate")
    live = _non_terminal_child(node_id=2, value_white=0.2)

    parent = _build_parent_eval({"loss": loss, "live": live})
    parent.set_evaluation(0.0)
    parent.tree_node.state.turn = Color.WHITE

    newly_over = _update_over(parent, {"loss"})

    assert not newly_over
    assert not parent.over_event.is_over()


def test_update_over_turn_sensitivity_win_for_opponent_does_not_force_over() -> None:
    white_win = _terminal_child(node_id=1, winner=Color.WHITE, term="white_mate")
    live = _non_terminal_child(node_id=2, value_white=0.2)

    parent = _build_parent_eval({"w": white_win, "live": live})
    parent.set_evaluation(0.0)
    parent.tree_node.state.turn = Color.BLACK

    newly_over = _update_over(parent, {"w"})

    assert not newly_over
    assert not parent.over_event.is_over()


def test_update_over_idempotent_second_call_no_change() -> None:
    win = _terminal_child(node_id=1, winner=Color.WHITE, term="mate")
    parent = _build_parent_eval({"win": win})
    parent.set_evaluation(0.0)
    parent.tree_node.state.turn = Color.WHITE

    first = _update_over(parent, {"win"})
    second = _update_over(parent, {"win"})

    assert first
    assert not second
    assert parent.over_event.is_over()
    assert parent.over_event.termination == "mate"
