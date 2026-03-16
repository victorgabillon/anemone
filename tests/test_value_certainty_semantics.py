"""Focused tests for exactness versus terminality Value semantics."""

from dataclasses import dataclass

from valanga.evaluations import Certainty, Value

from anemone.node_evaluation.common.canonical_value import (
    ValueSemanticsError,
    has_over_event,
    is_exact_value,
    is_forced_value,
    is_terminal_value,
    validate_value_semantics,
)


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
        over_event=None,
    )

    assert is_terminal_value(terminal)
    assert not is_terminal_value(forced)
    assert not is_terminal_value(estimate)


def test_over_event_helper_tracks_metadata_without_implying_terminality() -> None:
    forced_without_over = Value(score=0.0, certainty=Certainty.FORCED)
    forced_with_over = Value(
        score=0.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(),
    )
    terminal_with_over = Value(
        score=0.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(),
    )

    assert not has_over_event(forced_without_over)
    assert has_over_event(forced_with_over)
    assert has_over_event(terminal_with_over)


def test_value_semantics_validation_rejects_invalid_over_event_combinations() -> None:
    try:
        validate_value_semantics(
            Value(
                score=0.0,
                certainty=Certainty.ESTIMATE,
                over_event=_FakeOverEvent(),
            )
        )
    except ValueSemanticsError as exc:
        assert "ESTIMATE" in str(exc)
    else:
        raise AssertionError

    try:
        validate_value_semantics(
            Value(score=0.0, certainty=Certainty.TERMINAL, over_event=None)
        )
    except ValueSemanticsError as exc:
        assert "TERMINAL" in str(exc)
    else:
        raise AssertionError
