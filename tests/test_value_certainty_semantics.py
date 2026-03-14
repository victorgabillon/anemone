"""Focused tests for exactness versus terminality Value semantics."""

from dataclasses import dataclass

from anemone.node_evaluation.canonical_value import (
    is_exact_value,
    is_forced_value,
    is_terminal_candidate_value,
    is_terminal_value,
)
from valanga.evaluations import Certainty, Value


@dataclass(frozen=True)
class _FakeOverEvent:
    def is_over(self) -> bool:
        return True

    def is_draw(self) -> bool:
        return False

    def is_winner(self, player: object) -> bool:
        del player
        return False


def test_exact_value_helper_distinguishes_estimate_forced_terminal() -> None:
    estimate = Value(score=0.1, certainty=Certainty.ESTIMATE)
    forced = Value(score=0.2, certainty=Certainty.FORCED)
    terminal = Value(
        score=0.3,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(),
    )

    assert not is_exact_value(estimate)
    assert is_exact_value(forced)
    assert is_exact_value(terminal)
    assert is_forced_value(forced)
    assert not is_forced_value(terminal)


def test_terminal_value_helper_only_accepts_true_terminal() -> None:
    terminal = Value(
        score=0.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(),
    )
    forced = Value(
        score=0.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(),
    )
    estimate = Value(
        score=0.0,
        certainty=Certainty.ESTIMATE,
        over_event=_FakeOverEvent(),
    )

    assert is_terminal_value(terminal)
    assert not is_terminal_value(forced)
    assert not is_terminal_value(estimate)


def test_legacy_terminal_candidate_helper_accepts_forced_with_over_event() -> None:
    forced_without_over = Value(score=0.0, certainty=Certainty.FORCED)
    forced_with_over = Value(
        score=0.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(),
    )
    terminal_without_over = Value(score=0.0, certainty=Certainty.TERMINAL)
    estimate_with_over = Value(
        score=0.0,
        certainty=Certainty.ESTIMATE,
        over_event=_FakeOverEvent(),
    )

    assert not is_terminal_candidate_value(forced_without_over)
    assert is_terminal_candidate_value(forced_with_over)
    assert not is_terminal_candidate_value(terminal_without_over)
    assert not is_terminal_candidate_value(estimate_with_over)
