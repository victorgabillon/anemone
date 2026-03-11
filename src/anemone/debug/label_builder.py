"""Build human-readable labels for debug snapshots."""

from __future__ import annotations

from typing import Any

from .evaluation_inspectors.resolver import EvaluationDebugInspectorResolver
from .formatting import safe_getattr
from .index_formatter import format_index_lines


class NodeDebugLabelBuilder:
    """Build text labels for debug snapshots by inspecting node state."""

    def __init__(
        self,
        evaluation_resolver: EvaluationDebugInspectorResolver | None = None,
    ) -> None:
        """Initialize the label builder with an optional evaluation resolver."""
        self._evaluation_resolver = (
            evaluation_resolver or EvaluationDebugInspectorResolver()
        )

    def build_label(self, node: Any) -> str:
        """Return a multi-line label for ``node`` suitable for Graphviz."""
        node_id = safe_getattr(node, "id")
        depth = safe_getattr(node, "tree_depth")
        lines = [f"id={node_id}", f"depth={depth}"]

        state_tag = self._get_state_tag(node)
        if state_tag is not None:
            lines.append(f"state={state_tag}")

        evaluation = safe_getattr(node, "tree_evaluation")
        if evaluation is not None:
            lines.extend(self._evaluation_resolver.summarize(evaluation))

        lines.extend(self._get_index_lines(node))

        return "\n".join(lines)

    def _get_state_tag(self, node: Any) -> str | None:
        """Return the node state's tag when available."""
        state = safe_getattr(node, "state")
        if state is None:
            node_tag = safe_getattr(node, "tag")
            return None if node_tag is None else str(node_tag)

        tag = safe_getattr(state, "tag")
        if tag is None:
            node_tag = safe_getattr(node, "tag")
            return None if node_tag is None else str(node_tag)

        return str(tag)

    def _get_index_lines(self, node: Any) -> list[str]:
        """Collect exploration-index lines from ``node`` if present."""
        index_data = safe_getattr(node, "exploration_index_data")
        if index_data is None:
            return []

        return format_index_lines(index_data)
