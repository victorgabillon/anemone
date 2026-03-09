"""Small shared helpers for canonical Value access across node-evaluation families."""

from valanga import OverEvent

from anemone.values import Certainty, Value


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


def is_terminal_candidate_value(value: Value | None) -> bool:
    """Return True when a candidate Value is terminal/forced with over metadata."""
    return (
        value is not None
        and value.certainty in (Certainty.TERMINAL, Certainty.FORCED)
        and value.over_event is not None
    )


def get_over_event_candidate(value: Value | None) -> OverEvent | None:
    """Return terminal metadata from a candidate Value when present."""
    if value is None:
        return None
    return value.over_event
