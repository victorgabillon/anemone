"""Structured metadata extraction for debug snapshot nodes."""

# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .formatting import (
    format_branch_sequence,
    format_over_event,
    format_value,
    resolve_evaluation_over_event,
    safe_getattr,
)
from .index_formatter import format_index_fields


def _empty_string_dict() -> dict[str, str]:
    """Return an empty string-keyed dictionary for dataclass defaults."""
    return {}


@dataclass(frozen=True)
class NodeDebugMetadata:
    """Structured best-effort metadata extracted from a runtime node."""

    state_tag: str | None = None
    direct_value: str | None = None
    backed_up_value: str | None = None
    principal_variation: str | None = None
    over_event: str | None = None
    index_fields: dict[str, str] = field(default_factory=_empty_string_dict)


class NodeDebugMetadataBuilder:
    """Build structured debug metadata for a runtime node."""

    def build_metadata(self, node: Any) -> NodeDebugMetadata:
        """Return structured debug metadata for ``node``."""
        evaluation = safe_getattr(node, "tree_evaluation")
        return NodeDebugMetadata(
            state_tag=self._get_state_tag(node),
            direct_value=self._get_direct_value(evaluation),
            backed_up_value=self._get_backed_up_value(evaluation),
            principal_variation=self._get_principal_variation(evaluation),
            over_event=self._get_over_event(evaluation),
            index_fields=self._get_index_fields(node),
        )

    def _get_state_tag(self, node: Any) -> str | None:
        """Return the node state's tag when available."""
        state = safe_getattr(node, "state")
        if state is not None:
            tag = safe_getattr(state, "tag")
            if tag is not None:
                return str(tag)

        node_tag = safe_getattr(node, "tag")
        return None if node_tag is None else str(node_tag)

    def _get_direct_value(self, evaluation: Any) -> str | None:
        """Return a formatted direct value when available."""
        if evaluation is None:
            return None
        direct_value = safe_getattr(evaluation, "direct_value")
        return None if direct_value is None else format_value(direct_value)

    def _get_backed_up_value(self, evaluation: Any) -> str | None:
        """Return a formatted backed-up value when available."""
        if evaluation is None:
            return None

        backed_up_value = safe_getattr(evaluation, "backed_up_value")
        if backed_up_value is not None:
            return format_value(backed_up_value)

        minmax_value = safe_getattr(evaluation, "minmax_value")
        return None if minmax_value is None else format_value(minmax_value)

    def _get_principal_variation(self, evaluation: Any) -> str | None:
        """Return a formatted principal variation when available."""
        if evaluation is None:
            return None

        best_branch_sequence = safe_getattr(evaluation, "best_branch_sequence")
        if not best_branch_sequence:
            return None
        return format_branch_sequence(best_branch_sequence)

    def _get_over_event(self, evaluation: Any) -> str | None:
        """Return a formatted over-event when available."""
        if evaluation is None:
            return None

        over_event = resolve_evaluation_over_event(evaluation)
        return None if over_event is None else format_over_event(over_event)

    def _get_index_fields(self, node: Any) -> dict[str, str]:
        """Return structured exploration-index fields when available."""
        index_data = safe_getattr(node, "exploration_index_data")
        if index_data is None:
            return {}
        return format_index_fields(index_data)


__all__ = ["NodeDebugMetadata", "NodeDebugMetadataBuilder"]
