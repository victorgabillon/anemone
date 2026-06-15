"""Rollout expansion reporting primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RolloutStopReason(StrEnum):
    """Reasons a deterministic materialized rollout path stopped."""

    ACTION_SELECTOR_STOP = "action_selector_stop"
    MAX_EXTRA_STEPS = "max_extra_steps"
    TERMINAL = "terminal"
    NO_LEGAL_ACTIONS = "no_legal_actions"
    EXISTING_NODE = "existing_node"
    BRANCH_BUDGET_EXHAUSTED = "branch_budget_exhausted"


def _new_stop_reason_counts() -> dict[str, int]:
    """Return a fresh stop-reason counter."""
    return {reason.value: 0 for reason in RolloutStopReason}


@dataclass(frozen=True, slots=True)
class RolloutPathReport:
    """Summary of one materialized rollout trajectory.

    ``total_edge_count`` counts materialized edges only:
    ``initial_edge_count + extra_edge_count``. Traversed already-opened edges are
    reported separately in ``traversal_count``.
    """

    start_node_id: str | None
    start_depth: int | None
    end_node_id: str | None
    end_depth: int | None
    initial_edge_count: int
    extra_edge_count: int
    traversal_count: int
    total_edge_count: int
    stop_reason: str
    end_is_terminal: bool | None
    end_is_exact: bool | None


@dataclass(frozen=True, slots=True)
class RolloutExpansionReport:
    """Summary of one rollout expansion batch."""

    path_count: int = 0
    initial_edge_count: int = 0
    extra_edge_count: int = 0
    traversal_count: int = 0
    total_edge_count: int = 0
    created_node_count: int = 0
    existing_node_stop_count: int = 0
    max_extra_depth_reached: int = 0
    stop_reason_counts: dict[str, int] = field(default_factory=_new_stop_reason_counts)
    path_reports: tuple[RolloutPathReport, ...] = ()


@dataclass(slots=True)
class RolloutExpansionReportBuilder:
    """Mutable builder for a rollout expansion report."""

    path_count: int = 0
    initial_edge_count: int = 0
    extra_edge_count: int = 0
    traversal_count: int = 0
    created_node_count: int = 0
    existing_node_stop_count: int = 0
    max_extra_depth_reached: int = 0
    stop_reason_counts: dict[str, int] = field(default_factory=_new_stop_reason_counts)
    path_reports: list[RolloutPathReport] = field(default_factory=list)

    def record_initial_edge(self, *, created_node: bool) -> None:
        """Record an initial instruction edge."""
        self.path_count += 1
        self.initial_edge_count += 1
        if created_node:
            self.created_node_count += 1

    def record_extra_edge(
        self,
        *,
        rollout_depth: int,
        created_node: bool,
    ) -> None:
        """Record a rollout continuation edge."""
        self.extra_edge_count += 1
        self.max_extra_depth_reached = max(
            self.max_extra_depth_reached,
            rollout_depth,
        )
        if created_node:
            self.created_node_count += 1

    def record_traversal(self) -> None:
        """Record traversal through an already-opened rollout edge."""
        self.traversal_count += 1

    def record_stop(self, reason: RolloutStopReason) -> None:
        """Record one path stop reason."""
        self.stop_reason_counts[reason.value] += 1
        if reason is RolloutStopReason.EXISTING_NODE:
            self.existing_node_stop_count += 1

    def record_path(self, path_report: RolloutPathReport) -> None:
        """Record one per-path rollout report."""
        self.path_reports.append(path_report)

    def build(self) -> RolloutExpansionReport:
        """Build an immutable report snapshot."""
        total_edge_count = self.initial_edge_count + self.extra_edge_count
        return RolloutExpansionReport(
            path_count=self.path_count,
            initial_edge_count=self.initial_edge_count,
            extra_edge_count=self.extra_edge_count,
            traversal_count=self.traversal_count,
            total_edge_count=total_edge_count,
            created_node_count=self.created_node_count,
            existing_node_stop_count=self.existing_node_stop_count,
            max_extra_depth_reached=self.max_extra_depth_reached,
            stop_reason_counts=dict(self.stop_reason_counts),
            path_reports=tuple(self.path_reports),
        )


__all__ = [
    "RolloutExpansionReport",
    "RolloutExpansionReportBuilder",
    "RolloutPathReport",
    "RolloutStopReason",
]
