"""Selection report dataclasses and text formatting for Linoo."""

# pylint: disable=duplicate-code

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .types import LinooDepthSelectionPolicy


@dataclass(frozen=True, slots=True)
class LinooDepthSelectionRow:
    """Selector-observable summary for one depth in the live tree."""

    depth: int
    total_nodes: int
    opened_count: int
    frontier_count: int
    terminal_count: int
    exact_count: int
    uncached_terminal_candidates: int
    non_openable_count: int
    selection_index: int | None
    active: bool
    selected: bool
    selection_weight: float | None = None
    selection_probability: float | None = None

    def __post_init__(self) -> None:
        """Keep the diagnostic accounting honest in debug runs."""
        assert self.total_nodes == (
            self.opened_count
            + self.frontier_count
            + self.terminal_count
            + self.exact_count
            + self.uncached_terminal_candidates
            + self.non_openable_count
        )


@dataclass(frozen=True, slots=True)
class LinooSelectionReport:
    """Structured observability payload for the latest Linoo selection."""

    selected_depth: int
    selected_node_id: int
    selected_node_direct_value: float | None
    selected_node_candidate_value: float | None
    selected_node_priority: float | None
    selected_node_rank: int
    ranked_candidate_count: int
    node_selection_policy: Literal["zipf_rank"]
    depth_selection_policy: LinooDepthSelectionPolicy
    selected_depth_selection_index: int
    depth_rows: tuple[LinooDepthSelectionRow, ...]
    selected_depth_selection_weight: float | None = None
    selected_depth_selection_probability: float | None = None
    collect_frontier_state_s: float | None = None
    choose_depth_s: float | None = None
    heap_update_s: float | None = None
    choose_node_s: float | None = None
    make_report_s: float | None = None
    total_s: float | None = None
    depth_row_count: int | None = None
    total_nodes_scanned: int | None = None
    frontier_nodes_scanned: int | None = None
    uncached_terminal_candidates: int | None = None
    selected_depth_frontier_count: int | None = None
    stale_candidates_skipped: int | None = None
    heap_candidates_registered: int | None = None
    heap_update_candidate_count: int | None = None
    heap_update_push_count: int | None = None
    heap_update_pop_count: int | None = None
    heap_update_stale_skip_count: int | None = None
    heap_update_signature_check_count: int | None = None
    heap_update_signature_recompute_count: int | None = None
    heap_update_version_mismatch_count: int | None = None
    heap_update_priority_state_free_count: int | None = None
    heap_update_priority_stateful_fallback_count: int | None = None
    heap_update_candidate_direct_count: int | None = None
    heap_update_candidate_tree_count: int | None = None
    heap_update_candidate_unknown_count: int | None = None
    heap_update_total_heap_entries: int | None = None
    heap_update_max_heap_size: int | None = None
    heap_update_depth_count: int | None = None
    heap_update_frontier_node_count_seen: int | None = None
    state_rebuilt: bool | None = None
    nodes_incrementally_updated: int | None = None

    def format_depth_table(self) -> str:
        """Return an aligned text table for Linoo depth diagnostics."""
        headers = (
            "depth",
            "total",
            "opened",
            "frontier",
            "terminal",
            "exact",
            "uncached_terminal",
            "non_openable",
            "index",
            "weight",
            "probability",
            "selected",
        )
        rows: tuple[tuple[str, ...], ...] = tuple(
            (
                str(row.depth),
                str(row.total_nodes),
                str(row.opened_count),
                str(row.frontier_count),
                str(row.terminal_count),
                str(row.exact_count),
                str(row.uncached_terminal_candidates),
                str(row.non_openable_count),
                _format_optional_int(row.selection_index),
                _format_optional_float(row.selection_weight),
                _format_optional_float(row.selection_probability),
                "yes" if row.selected else "no",
            )
            for row in self.depth_rows
        )
        return _format_table(headers, rows)


def _format_optional_int(value: int | None) -> str:
    """Format an optional integer for compact table display."""
    return "-" if value is None else str(value)


def _format_optional_float(value: float | None) -> str:
    """Format an optional float for compact table display."""
    return "-" if value is None else f"{value:.6g}"


def _format_table(
    headers: tuple[str, ...],
    rows: Sequence[Sequence[str]],
) -> str:
    """Format string rows as a simple aligned whitespace table."""
    widths = [
        max([len(headers[column]), *(len(row[column]) for row in rows)])
        for column in range(len(headers))
    ]
    formatted_rows = [
        " ".join(value.rjust(widths[column]) for column, value in enumerate(row))
        for row in rows
    ]
    return "\n".join(
        [
            " ".join(
                header.rjust(widths[column]) for column, header in enumerate(headers)
            ),
            *formatted_rows,
        ]
    )
