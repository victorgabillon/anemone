"""Ordering utilities for Value comparison and search sorting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from valanga import Color
from valanga.evaluations import Certainty, Value

if TYPE_CHECKING:
    from valanga import OverEvent


class TerminalOutcome(Enum):
    """Terminal outcomes from the side-to-move perspective."""

    LOSS = auto()
    DRAW = auto()
    WIN = auto()


@dataclass(frozen=True, slots=True)
class EvaluationOrdering:
    """Define terminal projection and exact-Value comparison semantics."""

    win_score: float = 1.0
    draw_score: float = 0.0
    loss_score: float = -1.0
    terminal_without_over_event: TerminalOutcome = TerminalOutcome.DRAW

    def terminal_score(self, over_event: OverEvent, *, perspective: Color) -> float:
        """Project a terminal over-event to a scalar score."""
        if over_event.is_draw():
            return self.draw_score
        if over_event.is_win_for(perspective):
            return self.win_score
        return self.loss_score

    def semantic_compare(self, a: Value, b: Value, *, side_to_move: Color) -> int:
        """Compare values with terminal-outcome and exactness-aware rules."""
        a_outcome = self._terminal_outcome_or_none(a, perspective=side_to_move)
        b_outcome = self._terminal_outcome_or_none(b, perspective=side_to_move)

        if a_outcome is not None and b_outcome is not None:
            outcome_comparison = _compare_terminal_outcomes(a_outcome, b_outcome)
            if outcome_comparison != 0:
                return outcome_comparison

            # Exact values with the same terminal outcome still need their raw
            # solved score to break ties. Otherwise two exact wins/losses look
            # identical and branch order falls back to insertion order.
            return _compare_scores(a.score, b.score, side_to_move=side_to_move)

        if a_outcome is not None:
            assert b_outcome is None
            return self._terminal_vs_estimate(a_outcome, b, side_to_move=side_to_move)

        if b_outcome is not None:
            assert a_outcome is None
            return -self._terminal_vs_estimate(b_outcome, a, side_to_move=side_to_move)

        return _compare_scores(a.score, b.score, side_to_move=side_to_move)

    def search_sort_key(
        self,
        value: Value,
        *,
        side_to_move: Color,
    ) -> float:
        """Return B-ish search key based on numeric projection.

        Uses projected numeric scores for terminal values and raw score for estimates,
        preserving legacy behavior where large-magnitude estimates can outrank
        projected terminal outcomes.
        """
        projected_score = self._projected_score(value, perspective=side_to_move)
        if side_to_move == Color.WHITE:
            return -projected_score
        return projected_score

    def _projected_score(self, value: Value, *, perspective: Color) -> float:
        outcome = self._terminal_outcome_or_none(value, perspective=perspective)
        if outcome is None:
            return value.score
        if outcome is TerminalOutcome.WIN:
            return self.win_score
        if outcome is TerminalOutcome.DRAW:
            return self.draw_score
        return self.loss_score

    def _terminal_outcome_or_none(
        self,
        value: Value,
        *,
        perspective: Color,
    ) -> TerminalOutcome | None:
        if not self._is_exact_value_for_ordering(value):
            return None

        over_event = value.over_event
        if over_event is None:
            return self.terminal_without_over_event

        if over_event.is_draw():
            return TerminalOutcome.DRAW
        if over_event.is_win_for(perspective):
            return TerminalOutcome.WIN
        return TerminalOutcome.LOSS

    def _is_exact_value_for_ordering(self, value: Value) -> bool:
        """Return True when ordering should treat the Value as exact.

        This is an ordering-only notion: `FORCED` participates here even though
        it does not mean the node's own state is terminal.
        """
        return value.certainty in {Certainty.TERMINAL, Certainty.FORCED}

    def _terminal_vs_estimate(
        self,
        terminal_outcome: TerminalOutcome,
        estimate: Value,
        *,
        side_to_move: Color,
    ) -> int:
        if terminal_outcome is TerminalOutcome.WIN:
            return 1
        if terminal_outcome is TerminalOutcome.LOSS:
            return -1
        draw_value = Value(score=self.draw_score, certainty=Certainty.FORCED)
        return _compare_scores(
            draw_value.score, estimate.score, side_to_move=side_to_move
        )


def _compare_scores(a_score: float, b_score: float, *, side_to_move: Color) -> int:
    if side_to_move == Color.WHITE:
        if a_score > b_score:
            return 1
        if a_score < b_score:
            return -1
        return 0

    if a_score < b_score:
        return 1
    if a_score > b_score:
        return -1
    return 0


def _compare_terminal_outcomes(a: TerminalOutcome, b: TerminalOutcome) -> int:
    rank = {
        TerminalOutcome.LOSS: 0,
        TerminalOutcome.DRAW: 1,
        TerminalOutcome.WIN: 2,
    }
    if rank[a] > rank[b]:
        return 1
    if rank[a] < rank[b]:
        return -1
    return 0


DEFAULT_EVALUATION_ORDERING = EvaluationOrdering()
