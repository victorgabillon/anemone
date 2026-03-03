"""Unit tests for EvaluationOrdering semantics."""

from dataclasses import dataclass

from valanga import Color

from anemone.values import Certainty, EvaluationOrdering, Value


@dataclass(frozen=True)
class _FakeOverEvent:
    winner: Color | None = None
    draw: bool = False

    def is_over(self) -> bool:
        return self.draw or self.winner is not None

    def is_draw(self) -> bool:
        return self.draw

    def is_winner(self, player: Color) -> bool:
        return self.winner == player


def test_semantic_compare_win_beats_high_estimate() -> None:
    ordering = EvaluationOrdering(win_score=1.0, draw_score=0.0, loss_score=-1.0)
    win = Value(
        score=-10.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    estimate_high = Value(score=100.0)
    assert ordering.semantic_compare(win, estimate_high, side_to_move=Color.WHITE) > 0


def test_semantic_compare_loss_loses_to_estimate() -> None:
    ordering = EvaluationOrdering()
    loss = Value(
        score=10.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(winner=Color.BLACK),
    )
    estimate_low = Value(score=-0.9)
    assert ordering.semantic_compare(loss, estimate_low, side_to_move=Color.WHITE) < 0


def test_semantic_compare_draw_vs_estimate_uses_draw_score_for_both_sides() -> None:
    ordering = EvaluationOrdering(win_score=1.0, draw_score=0.0, loss_score=-1.0)
    draw = Value(
        score=99.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(draw=True),
    )

    white_below_draw = Value(score=-0.1)
    white_above_draw = Value(score=0.1)
    assert ordering.semantic_compare(draw, white_below_draw, side_to_move=Color.WHITE) > 0
    assert ordering.semantic_compare(draw, white_above_draw, side_to_move=Color.WHITE) < 0

    black_below_draw = Value(score=-0.1)
    black_above_draw = Value(score=0.1)
    assert ordering.semantic_compare(draw, black_below_draw, side_to_move=Color.BLACK) < 0
    assert ordering.semantic_compare(draw, black_above_draw, side_to_move=Color.BLACK) > 0


def test_search_sort_key_is_projection_based_bish() -> None:
    ordering = EvaluationOrdering(win_score=1.0, draw_score=0.0, loss_score=-1.0)
    win = Value(
        score=-100.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )

    white_estimate_higher = Value(score=2.0)
    assert ordering.search_sort_key(white_estimate_higher, side_to_move=Color.WHITE) < (
        ordering.search_sort_key(win, side_to_move=Color.WHITE)
    )

    black_win = Value(
        score=100.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.BLACK),
    )
    black_estimate_lower = Value(score=-2.0)
    assert ordering.search_sort_key(black_estimate_lower, side_to_move=Color.BLACK) < (
        ordering.search_sort_key(black_win, side_to_move=Color.BLACK)
    )
