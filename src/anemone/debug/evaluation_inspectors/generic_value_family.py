"""Inspector for canonical value-family evaluation objects."""

from __future__ import annotations

from typing import Any

from anemone.debug.formatting import (
    format_branch_sequence,
    format_over_event,
    format_value,
    safe_getattr,
)


class GenericValueFamilyInspector:
    """Inspect evaluation families that expose canonical direct/backed-up values."""

    def supports(self, evaluation: Any) -> bool:
        """Return whether ``evaluation`` exposes canonical value-family fields."""
        return any(
            safe_getattr(evaluation, attribute_name) is not None
            for attribute_name in (
                "backed_up_value",
                "over_event",
            )
        )

    def summarize(self, evaluation: Any) -> list[str]:
        """Return human-readable debug lines for canonical value-family fields."""
        lines: list[str] = []

        direct_value = safe_getattr(evaluation, "direct_value")
        if direct_value is not None:
            lines.append(f"direct={format_value(direct_value)}")

        backed_up_value = safe_getattr(evaluation, "backed_up_value")
        if backed_up_value is not None:
            lines.append(f"backed_up={format_value(backed_up_value)}")

        best_branch_sequence = safe_getattr(evaluation, "best_branch_sequence")
        if best_branch_sequence:
            lines.append(f"pv={format_branch_sequence(best_branch_sequence)}")

        over_event = safe_getattr(evaluation, "over_event")
        if over_event is not None:
            lines.append(f"over={format_over_event(over_event)}")

        return lines
