"""Build human-readable labels for debug snapshots."""

from __future__ import annotations

from typing import Any


class NodeDebugLabelBuilder:
    """Build text labels for debug snapshots by inspecting node state."""

    def build_label(self, node: Any) -> str:
        """Return a multi-line label for ``node`` suitable for Graphviz."""
        lines = [f"id={node.id}", f"depth={node.tree_depth}"]

        state_tag = self._get_state_tag(node)
        if state_tag is not None:
            lines.append(f"state={state_tag}")

        lines.extend(self._get_evaluation_lines(node))
        lines.extend(self._get_index_lines(node))

        return "\n".join(lines)

    def _get_state_tag(self, node: Any) -> str | None:
        """Return the node state's tag when available."""
        state = self._safe_getattr(node, "state")
        if state is None:
            node_tag = self._safe_getattr(node, "tag")
            return None if node_tag is None else str(node_tag)

        tag = self._safe_getattr(state, "tag")
        if tag is None:
            node_tag = self._safe_getattr(node, "tag")
            return None if node_tag is None else str(node_tag)

        return str(tag)

    def _get_evaluation_lines(self, node: Any) -> list[str]:
        """Collect evaluation-related lines from ``node`` if present."""
        lines: list[str] = []
        evaluation = self._safe_getattr(node, "tree_evaluation")
        if evaluation is None:
            return lines

        direct_value = self._safe_getattr(evaluation, "direct_value")
        if direct_value is not None:
            lines.append(f"direct={self._format_value(direct_value)}")

        backed_up_value = self._safe_getattr(evaluation, "backed_up_value")
        if backed_up_value is None:
            backed_up_value = self._safe_getattr(evaluation, "minmax_value")
        if backed_up_value is not None:
            lines.append(f"backed_up={self._format_value(backed_up_value)}")

        best_branch_sequence = self._safe_getattr(evaluation, "best_branch_sequence")
        if best_branch_sequence:
            lines.append(f"pv={self._format_branch_sequence(best_branch_sequence)}")

        over_event = self._safe_getattr(evaluation, "over_event")
        if over_event is None:
            over_event = self._get_over_event_candidate(evaluation)
        if over_event is not None:
            lines.append(f"over={self._format_over_event(over_event)}")

        return lines

    def _get_index_lines(self, node: Any) -> list[str]:
        """Collect exploration-index lines from ``node`` if present."""
        index_data = self._safe_getattr(node, "exploration_index_data")
        if index_data is None:
            return []

        lines: list[str] = []
        for attribute_name, label in (
            ("index", "index"),
            ("zipf_factored_proba", "zipf_factored_proba"),
            ("min_path_value", "min_path_value"),
            ("max_path_value", "max_path_value"),
            ("max_depth_descendants", "max_depth_descendants"),
        ):
            value = self._safe_getattr(index_data, attribute_name)
            if value is not None:
                lines.append(f"{label}={value}")

        interval = self._safe_getattr(index_data, "interval")
        if interval is not None:
            min_value = self._safe_getattr(interval, "min_value")
            max_value = self._safe_getattr(interval, "max_value")
            if min_value is not None or max_value is not None:
                lines.append(f"interval=[{min_value}, {max_value}]")

        if lines:
            return lines

        return [f"index={index_data}"]

    def _get_over_event_candidate(self, evaluation: Any) -> Any | None:
        """Safely call ``get_over_event_candidate`` when an evaluation exposes it."""
        getter = self._safe_getattr(evaluation, "get_over_event_candidate")
        if not callable(getter):
            return None

        try:
            return getter()
        except (AttributeError, TypeError):
            return None

    def _format_value(self, value: Any) -> str:
        """Render a compact string for a value-like object."""
        score = self._safe_getattr(value, "score")
        certainty = self._safe_getattr(value, "certainty")
        over_event = self._safe_getattr(value, "over_event")

        parts: list[str] = []
        if score is not None:
            parts.append(f"score={score}")
        if certainty is not None:
            certainty_name = self._safe_getattr(certainty, "name")
            parts.append(
                f"certainty={certainty_name if certainty_name is not None else certainty}"
            )
        if over_event is not None:
            parts.append(f"over={self._format_over_event(over_event)}")

        return ", ".join(parts) if parts else str(value)

    def _format_branch_sequence(self, sequence: Any) -> str:
        """Render a principal-variation sequence."""
        try:
            return " -> ".join(str(item) for item in sequence)
        except TypeError:
            return str(sequence)

    def _format_over_event(self, over_event: Any) -> str:
        """Render an over-event-like object."""
        get_over_tag = self._safe_getattr(over_event, "get_over_tag")
        if callable(get_over_tag):
            try:
                return str(get_over_tag())
            except (AttributeError, TypeError):
                return str(over_event)
        return str(over_event)

    def _safe_getattr(self, obj: Any, attribute_name: str) -> Any | None:
        """Return ``obj.attribute_name`` while tolerating missing or fragile debug data."""
        try:
            return getattr(obj, attribute_name)
        except (AttributeError, TypeError):
            return None
