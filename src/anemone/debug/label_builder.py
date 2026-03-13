"""Build human-readable labels for debug snapshots."""

from __future__ import annotations

from typing import Any

from .formatting import safe_getattr
from .node_metadata_builder import NodeDebugMetadataBuilder


class NodeDebugLabelBuilder:
    """Build text labels for debug snapshots by inspecting node state."""

    def __init__(
        self,
        evaluation_resolver: object | None = None,
        metadata_builder: NodeDebugMetadataBuilder | None = None,
    ) -> None:
        """Initialize the label builder with an optional metadata builder."""
        # Kept only for compatibility with older construction sites; label
        # generation now flows through ``metadata_builder`` instead.
        del evaluation_resolver
        self._metadata_builder = metadata_builder or NodeDebugMetadataBuilder()

    def build_label(self, node: Any) -> str:
        """Return a multi-line label for ``node`` suitable for Graphviz."""
        metadata = self._metadata_builder.build_metadata(node)
        node_id = safe_getattr(node, "id")
        depth = safe_getattr(node, "tree_depth")
        lines = [f"id={node_id}", f"depth={depth}"]

        if metadata.player_label is not None:
            lines.append(f"player={metadata.player_label}")
        if metadata.state_tag is not None:
            lines.append(f"state={metadata.state_tag}")
        if metadata.direct_value is not None:
            lines.append(f"direct={metadata.direct_value}")
        if metadata.backed_up_value is not None:
            lines.append(f"backed_up={metadata.backed_up_value}")
        if metadata.principal_variation is not None:
            lines.append(f"pv={metadata.principal_variation}")
        if metadata.over_event is not None:
            lines.append(f"over={metadata.over_event}")
        lines.extend(
            f"{field_name}={field_value}"
            for field_name, field_value in metadata.index_fields.items()
        )

        return "\n".join(lines)
