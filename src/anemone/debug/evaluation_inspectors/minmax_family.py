"""Compatibility inspector for minmax-style evaluation objects."""

from __future__ import annotations

from typing import Any

from anemone.debug.formatting import (
    format_branch_sequence,
    format_over_event,
    format_value,
    resolve_evaluation_over_event,
    safe_getattr,
    safe_hasattr,
)


class MinmaxFamilyInspector:
    """Inspect minmax-style evaluation objects that expose compatibility fields."""

    def supports(self, evaluation: Any) -> bool:
        """Return whether ``evaluation`` exposes minmax compatibility fields."""
        if safe_hasattr(evaluation, "minmax_value"):
            return True
        getter = safe_getattr(evaluation, "get_over_event_candidate")
        return callable(getter)

    def summarize(self, evaluation: Any) -> list[str]:
        """Return human-readable debug lines for minmax compatibility fields."""
        lines: list[str] = []

        direct_value = safe_getattr(evaluation, "direct_value")
        if direct_value is not None:
            lines.append(f"direct={format_value(direct_value)}")

        minmax_value = safe_getattr(evaluation, "minmax_value")
        if minmax_value is not None:
            lines.append(f"backed_up={format_value(minmax_value)}")

        best_branch_sequence = safe_getattr(evaluation, "best_branch_sequence")
        if best_branch_sequence:
            lines.append(f"pv={format_branch_sequence(best_branch_sequence)}")

        over_event = resolve_evaluation_over_event(evaluation)
        if over_event is not None:
            lines.append(f"over={format_over_event(over_event)}")

        return lines
