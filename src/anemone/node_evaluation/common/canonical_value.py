"""Shared helpers for canonical Value access and certainty semantics.

`Certainty` carries two related but distinct ideas:

- `TERMINAL`: the node's own state is over and the value is exact.
- `FORCED`: the node is not terminal, but its backed-up value is exact.
- `ESTIMATE`: the value is still heuristic or otherwise not fully solved.

The helpers below keep exactness checks separate from state-terminality checks so
callers can be explicit about which notion they need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from valanga.evaluations import Certainty, Value

if TYPE_CHECKING:
    from anemone._valanga_types import AnyOverEvent


class ValueSemanticsError(ValueError):
    """Raised when a Value violates Anemone's certainty/metadata contract."""

    @classmethod
    def missing_value(cls) -> ValueSemanticsError:
        """Return the error for missing Value objects."""
        return cls("Expected a Value, got None.")

    @classmethod
    def estimate_with_over_event(cls) -> ValueSemanticsError:
        """Return the error for ESTIMATE values carrying terminal metadata."""
        return cls("ESTIMATE values must not carry over_event metadata.")

    @classmethod
    def terminal_without_over_event(cls) -> ValueSemanticsError:
        """Return the error for TERMINAL values missing terminal metadata."""
        return cls("TERMINAL values must carry over_event metadata.")

    @classmethod
    def non_over_over_event(cls) -> ValueSemanticsError:
        """Return the error for invalid over-event metadata."""
        return cls("Values with over_event metadata must point to an over outcome.")


def validate_value_semantics(value: Value) -> Value:
    """Validate a Value against Anemone's certainty semantics."""
    if value.certainty == Certainty.ESTIMATE and value.over_event is not None:
        raise ValueSemanticsError.estimate_with_over_event()

    if value.certainty == Certainty.TERMINAL and value.over_event is None:
        raise ValueSemanticsError.terminal_without_over_event()

    if value.over_event is not None and not value.over_event.is_over():
        raise ValueSemanticsError.non_over_over_event()

    return value


def require_value(value: Value | None) -> Value:
    """Return ``value`` once presence and semantics are both validated."""
    if value is None:
        raise ValueSemanticsError.missing_value()
    return validate_value_semantics(value)


def _validated_optional_value(value: Value | None) -> Value | None:
    """Return ``value`` after semantic validation when present."""
    if value is None:
        return None
    return validate_value_semantics(value)


def get_value_candidate(
    *,
    backed_up_value: Value | None,
    direct_value: Value | None,
) -> Value | None:
    """Return backed-up value when available, else direct value."""
    if backed_up_value is not None:
        return validate_value_semantics(backed_up_value)
    return _validated_optional_value(direct_value)


def get_value(
    *,
    backed_up_value: Value | None,
    direct_value: Value | None,
) -> Value:
    """Return the canonical Value for a node-evaluation family."""
    return require_value(
        get_value_candidate(
            backed_up_value=backed_up_value,
            direct_value=direct_value,
        )
    )


def get_score(
    *,
    backed_up_value: Value | None,
    direct_value: Value | None,
) -> float:
    """Return the canonical scalar score for a node-evaluation family."""
    return get_value(
        backed_up_value=backed_up_value,
        direct_value=direct_value,
    ).score


def is_exact_value(value: Value | None) -> bool:
    """Return True when a Value is exact, whether forced or terminal."""
    value = _validated_optional_value(value)
    return value is not None and value.certainty in (
        Certainty.FORCED,
        Certainty.TERMINAL,
    )


def is_estimate_value(value: Value | None) -> bool:
    """Return True when a Value is still heuristic or otherwise unsolved."""
    value = _validated_optional_value(value)
    return value is not None and value.certainty == Certainty.ESTIMATE


def is_forced_value(value: Value | None) -> bool:
    """Return True when a Value is exact for a non-terminal node."""
    value = _validated_optional_value(value)
    return value is not None and value.certainty == Certainty.FORCED


def is_terminal_value(value: Value | None) -> bool:
    """Return True when a Value says the node's own state is terminal."""
    value = _validated_optional_value(value)
    return value is not None and value.certainty == Certainty.TERMINAL


def has_over_event(value: Value | None) -> bool:
    """Return True when a Value carries exact terminal-outcome metadata."""
    value = _validated_optional_value(value)
    return value is not None and value.over_event is not None


def make_estimate_value(*, score: float) -> Value:
    """Create an unsolved estimate Value."""
    return Value(
        score=score,
        certainty=Certainty.ESTIMATE,
        over_event=None,
    )


def make_forced_value(*, score: float, over_event: AnyOverEvent | None = None) -> Value:
    """Create an exact non-terminal Value."""
    return validate_value_semantics(
        Value(
            score=score,
            certainty=Certainty.FORCED,
            over_event=over_event,
        )
    )


def make_terminal_value(*, score: float, over_event: AnyOverEvent) -> Value:
    """Create an exact Value for a node whose own state is terminal."""
    return validate_value_semantics(
        Value(
            score=score,
            certainty=Certainty.TERMINAL,
            over_event=over_event,
        )
    )


def make_backed_up_value(
    *,
    score: float,
    exact: bool,
    node_is_terminal: bool,
    over_event: AnyOverEvent | None,
) -> Value:
    """Construct a backed-up Value from parent-local terminality and exactness.

    Important: exactness may propagate from children, but terminality never does.
    Only a node whose own state is terminal may receive ``Certainty.TERMINAL``.
    """
    if node_is_terminal:
        if over_event is None:
            raise ValueSemanticsError.terminal_without_over_event()
        return make_terminal_value(score=score, over_event=over_event)
    if exact:
        return make_forced_value(score=score, over_event=over_event)
    if over_event is not None:
        raise ValueSemanticsError.estimate_with_over_event()
    return make_estimate_value(score=score)


def get_over_event_candidate(value: Value | None) -> AnyOverEvent | None:
    """Return exact outcome metadata from a candidate Value when present."""
    validated_value = _validated_optional_value(value)
    if validated_value is None:
        return None
    return validated_value.over_event
