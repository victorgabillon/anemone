"""Shared Linoo selector types and status constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Runtime import is intentional: parsley/get_type_hints() evaluates LinooArgs
# annotations when Chipiron builds partial dataclasses.
from anemone.node_selector.node_selector_types import NodeSelectorType  # noqa: TC001

type LinooDepthSelectionPolicy = Literal[
    "inverse_depth",
    "opened_count_depth_index",
]


@dataclass
class LinooArgs:
    """Arguments for the Linoo node selector."""

    type: Literal[NodeSelectorType.LINOO]
    depth_selection_policy: LinooDepthSelectionPolicy = "inverse_depth"


type LinooNodeStatus = Literal[
    "opened",
    "frontier",
    "terminal",
    "exact",
    "uncached_terminal_candidate",
    "non_openable",
]


LINOO_NODE_STATUS_COUNTERS: dict[LinooNodeStatus, str] = {
    "opened": "opened_count",
    "frontier": "frontier_count",
    "terminal": "terminal_count",
    "exact": "exact_count",
    "uncached_terminal_candidate": "uncached_terminal_candidates",
    "non_openable": "non_openable_count",
}
LINOO_NODE_STATUSES = frozenset(LINOO_NODE_STATUS_COUNTERS)
LINOO_DEFAULT_NODE_STATUS: LinooNodeStatus = "opened"
