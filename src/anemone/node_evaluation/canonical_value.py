"""Shared helpers for canonical Value access and certainty semantics.

`Certainty` carries two related but distinct ideas:

- `TERMINAL`: the node's own state is over and the value is exact.
- `FORCED`: the node is not terminal, but its backed-up value is exact.
- `ESTIMATE`: the value is still heuristic or otherwise not fully solved.

The helpers below keep exactness checks separate from state-terminality checks so
callers can be explicit about which notion they need.
"""

from valanga import OverEvent
from valanga.evaluations import Certainty, Value


def get_value_candidate(
    *,
    backed_up_value: Value | None,
    direct_value: Value | None,
) -> Value | None:
    """Return backed-up value when available, else direct value."""
    if backed_up_value is not None:
        return backed_up_value
    return direct_value


def get_value(
    *,
    backed_up_value: Value | None,
    direct_value: Value | None,
) -> Value:
    """Return the canonical Value for a node-evaluation family."""
    candidate = get_value_candidate(
        backed_up_value=backed_up_value,
        direct_value=direct_value,
    )
    assert candidate is not None, (
        "Node has no canonical Value: both backed_up_value and direct_value are None."
    )
    return candidate


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
    return value is not None and value.certainty in (
        Certainty.FORCED,
        Certainty.TERMINAL,
    )


def is_forced_value(value: Value | None) -> bool:
    """Return True when a Value is exact for a non-terminal node."""
    return value is not None and value.certainty == Certainty.FORCED


def is_terminal_value(value: Value | None) -> bool:
    """Return True when a Value says the node's own state is terminal."""
    return value is not None and value.certainty == Certainty.TERMINAL


def is_terminal_candidate_value(value: Value | None) -> bool:
    """Legacy compatibility helper for exact values carrying terminal metadata.

    This helper does NOT mean the node's own state is terminal in the stronger
    semantic sense used by the certainty model. PR A preserves its behavior for
    existing callers that still rely on exact-value-plus-``over_event`` checks;
    later refactors can narrow or replace those callers where true node
    terminality is required.
    """
    if value is None:
        return False
    return is_exact_value(value) and value.over_event is not None


def get_over_event_candidate(value: Value | None) -> OverEvent | None:
    """Return terminal metadata from a candidate Value when present."""
    if value is None:
        return None
    return value.over_event
