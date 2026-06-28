"""Sparse runtime cache state for Linoo."""

# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass

from .types import (
    LINOO_DEFAULT_NODE_STATUS,
    LINOO_NODE_STATUS_COUNTERS,
    LinooNodeStatus,
)


@dataclass(init=False, slots=True)
class LinooDepthStats:
    """Mutable accounting bucket for one depth in the cached tree view."""

    total_nodes: int
    opened_count: int
    frontier_count: int
    terminal_count: int
    exact_count: int
    uncached_terminal_candidates: int
    non_openable_count: int

    def __init__(self) -> None:
        """Initialize empty accounting for one depth."""
        self.total_nodes = 0
        self.opened_count = 0
        self.frontier_count = 0
        self.terminal_count = 0
        self.exact_count = 0
        self.uncached_terminal_candidates = 0
        self.non_openable_count = 0

    def count_for(self, status: LinooNodeStatus) -> int:
        """Return the counter value for one cached node status."""
        count = getattr(self, LINOO_NODE_STATUS_COUNTERS[status])
        assert isinstance(count, int)
        return count

    def increment(self, status: LinooNodeStatus) -> None:
        """Add one node with ``status`` to this depth."""
        setattr(
            self,
            LINOO_NODE_STATUS_COUNTERS[status],
            self.count_for(status) + 1,
        )

    def decrement(self, status: LinooNodeStatus) -> None:
        """Remove one node with ``status`` from this depth."""
        next_count = self.count_for(status) - 1
        assert next_count >= 0
        setattr(self, LINOO_NODE_STATUS_COUNTERS[status], next_count)

    def empty(self) -> bool:
        """Return whether this depth no longer tracks any nodes."""
        return self.total_nodes == 0


@dataclass(frozen=True, slots=True)
class LinooNodeState:
    """Cached non-default Linoo classification for one tree node.

    Linoo classifies each live node by ``node_id``, relative ``depth``, and
    ``status``. The sparse runtime treats absence from the node-state cache as
    the default state: the current live node at its current depth with status
    ``"opened"``. That state contributes only aggregate depth accounting, so it
    does not need a per-node object for reads, checkpoint build, or restore.
    """

    node_id: int
    depth: int
    status: LinooNodeStatus

    def is_default(self) -> bool:
        """Return whether this state is equivalent to absent sparse state."""
        return self.status == LINOO_DEFAULT_NODE_STATUS
