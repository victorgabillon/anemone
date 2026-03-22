"""Unit tests for EvaluationOrdering semantics."""

from dataclasses import dataclass

from valanga import Color
from valanga.evaluations import Certainty, Value

from anemone.values import EvaluationOrdering


@dataclass(frozen=True)
class _FakeOverEvent:
    winner: Color | None = None
    draw: bool = False

    def is_over(self) -> bool:
        return self.draw or self.winner is not None

    def is_draw(self) -> bool:
        return self.draw

    def is_win_for(self, role: Color) -> bool:
        return self.winner == role

    def is_loss_for(self, role: Color) -> bool:
        return self.winner is not None and self.winner != role


@dataclass(frozen=True)
class _NoIsOverOverEvent:
    winner: Color | None = None
    draw: bool = False

    def is_over(self) -> bool:
        raise AssertionError(
            "search/decision ordering should not call over_event.is_over"
        )

    def is_draw(self) -> bool:
        return self.draw

    def is_win_for(self, role: Color) -> bool:
        return self.winner == role

    def is_loss_for(self, role: Color) -> bool:
        return self.winner is not None and self.winner != role


def test_semantic_compare_win_beats_high_estimate() -> None:
    ordering = EvaluationOrdering(win_score=1.0, draw_score=0.0, loss_score=-1.0)
    win = Value(
        score=-10.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    estimate_high = Value(score=100.0, certainty=Certainty.ESTIMATE)
    assert ordering.semantic_compare(win, estimate_high, side_to_move=Color.WHITE) > 0


def test_semantic_compare_loss_loses_to_estimate() -> None:
    ordering = EvaluationOrdering()
    loss = Value(
        score=10.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(winner=Color.BLACK),
    )
    estimate_low = Value(score=-0.9, certainty=Certainty.ESTIMATE)
    assert ordering.semantic_compare(loss, estimate_low, side_to_move=Color.WHITE) < 0


def test_semantic_compare_draw_vs_estimate_uses_draw_score_for_both_sides() -> None:
    ordering = EvaluationOrdering(win_score=1.0, draw_score=0.0, loss_score=-1.0)
    draw = Value(
        score=99.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(draw=True),
    )

    white_below_draw = Value(score=-0.1, certainty=Certainty.ESTIMATE)
    white_above_draw = Value(score=0.1, certainty=Certainty.ESTIMATE)
    assert (
        ordering.semantic_compare(draw, white_below_draw, side_to_move=Color.WHITE) > 0
    )
    assert (
        ordering.semantic_compare(draw, white_above_draw, side_to_move=Color.WHITE) < 0
    )

    black_below_draw = Value(score=-0.1, certainty=Certainty.ESTIMATE)
    black_above_draw = Value(score=0.1, certainty=Certainty.ESTIMATE)
    assert (
        ordering.semantic_compare(draw, black_below_draw, side_to_move=Color.BLACK) < 0
    )
    assert (
        ordering.semantic_compare(draw, black_above_draw, side_to_move=Color.BLACK) > 0
    )


def test_search_sort_key_is_projection_based_bish() -> None:
    ordering = EvaluationOrdering(win_score=1.0, draw_score=0.0, loss_score=-1.0)
    win = Value(
        score=-100.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )

    white_estimate_higher = Value(score=2.0, certainty=Certainty.ESTIMATE)
    assert ordering.search_sort_key(white_estimate_higher, side_to_move=Color.WHITE) < (
        ordering.search_sort_key(win, side_to_move=Color.WHITE)
    )

    black_win = Value(
        score=100.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.BLACK),
    )
    black_estimate_lower = Value(score=-2.0, certainty=Certainty.ESTIMATE)
    assert ordering.search_sort_key(black_estimate_lower, side_to_move=Color.BLACK) < (
        ordering.search_sort_key(black_win, side_to_move=Color.BLACK)
    )


def test_ordering_does_not_use_over_event_is_over() -> None:
    ordering = EvaluationOrdering()
    forced_win = Value(
        score=-42.0,
        certainty=Certainty.FORCED,
        over_event=_NoIsOverOverEvent(winner=Color.WHITE),
    )
    estimate = Value(score=0.5, certainty=Certainty.ESTIMATE)

    assert ordering.semantic_compare(forced_win, estimate, side_to_move=Color.WHITE) > 0
    assert ordering.search_sort_key(forced_win, side_to_move=Color.WHITE) == -1.0


def test_semantic_compare_same_exact_win_uses_score_tie_break_for_white() -> None:
    ordering = EvaluationOrdering()
    smaller_win = Value(
        score=2.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    larger_win = Value(
        score=4.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )

    assert (
        ordering.semantic_compare(
            larger_win,
            smaller_win,
            side_to_move=Color.WHITE,
        )
        > 0
    )


def test_semantic_compare_same_exact_loss_uses_score_tie_break_for_black() -> None:
    ordering = EvaluationOrdering()
    smaller_loss = Value(
        score=2.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    larger_loss = Value(
        score=9.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )

    assert (
        ordering.semantic_compare(
            smaller_loss,
            larger_loss,
            side_to_move=Color.BLACK,
        )
        > 0
    )
