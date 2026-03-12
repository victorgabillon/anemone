"""Shared low-level formatting helpers for debug output."""

from __future__ import annotations

from typing import Any


def safe_getattr(obj: Any, attribute_name: str) -> Any | None:
    """Return ``obj.attribute_name`` when available for best-effort debug output."""
    try:
        return getattr(obj, attribute_name)
    except (AttributeError, TypeError):
        return None


def safe_hasattr(obj: Any, attribute_name: str) -> bool:
    """Return whether ``obj`` exposes ``attribute_name`` for debug purposes."""
    try:
        getattr(obj, attribute_name)
    except (AttributeError, TypeError):
        return False
    return True


def format_over_event(over_event: Any) -> str:
    """Render an over-event-like object."""
    get_over_tag = safe_getattr(over_event, "get_over_tag")
    if callable(get_over_tag):
        try:
            return str(get_over_tag())
        except (AttributeError, TypeError):
            return str(over_event)
    return str(over_event)


def resolve_evaluation_over_event(evaluation: Any) -> Any | None:
    """Return the best-effort over-event exposed by ``evaluation``."""
    over_event = safe_getattr(evaluation, "over_event")
    if over_event is not None:
        return over_event

    getter = safe_getattr(evaluation, "get_over_event_candidate")
    if callable(getter):
        try:
            return getter()
        except (AttributeError, TypeError):
            return None

    return None


def format_branch_sequence(sequence: Any) -> str:
    """Render a principal-variation sequence."""
    try:
        return " -> ".join(str(item) for item in sequence)
    except TypeError:
        return str(sequence)


def format_value(value: Any) -> str:
    """Render a compact string for a value-like object."""
    score = safe_getattr(value, "score")
    certainty = safe_getattr(value, "certainty")
    over_event = safe_getattr(value, "over_event")

    parts: list[str] = []
    if score is not None:
        parts.append(f"score={score}")
    if certainty is not None:
        certainty_name = safe_getattr(certainty, "name")
        parts.append(
            f"certainty={certainty_name if certainty_name is not None else certainty}"
        )
    if over_event is not None:
        parts.append(f"over={format_over_event(over_event)}")

    return ", ".join(parts) if parts else str(value)
