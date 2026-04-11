"""Small best-effort helpers shared across debug and export code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable


def safe_getattr(obj: Any, attribute_name: str) -> Any | None:
    """Return ``obj.attribute_name`` when available."""
    try:
        return getattr(obj, attribute_name)
    except (AttributeError, TypeError):
        return None


def format_over_event(over_event: object) -> str:
    """Render an over-event-like object."""
    get_over_tag = safe_getattr(over_event, "get_over_tag")
    if callable(get_over_tag):
        try:
            return str(cast("Callable[[], object]", get_over_tag)())
        except (AttributeError, TypeError):
            return str(over_event)
    return str(over_event)


def resolve_evaluation_over_event(evaluation: object) -> object | None:
    """Return the best-effort over-event exposed by ``evaluation``."""
    over_event = safe_getattr(evaluation, "over_event")
    if over_event is not None:
        return cast("object", over_event)

    getter = safe_getattr(evaluation, "get_over_event_candidate")
    if callable(getter):
        try:
            return cast("Callable[[], object]", getter)()
        except (AttributeError, TypeError):
            return None

    return None


def coerce_int(value: object, *, default: int | None = None) -> int:
    """Return ``value`` as ``int`` for supported scalar payloads."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | str):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if default is not None:
        return default
    raise TypeError
