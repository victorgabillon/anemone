"""Depth-selection policies for Linoo."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import (
    invalid_linoo_depth_selection_policy_error,
    no_frontier_nodes_error,
)

if TYPE_CHECKING:
    from random import Random

    from .report import LinooDepthSelectionRow
    from .runtime_state import LinooDepthStats
    from .types import LinooDepthSelectionPolicy


def choose_depth(
    *,
    depth_selection_policy: LinooDepthSelectionPolicy,
    depth_stats_by_depth: dict[int, LinooDepthStats],
    active_depths: tuple[int, ...],
    random_generator: Random,
) -> int:
    """Choose one active depth using the configured Linoo policy."""
    if not active_depths:
        raise no_frontier_nodes_error()
    if depth_selection_policy == "opened_count_depth_index":
        return choose_depth_by_opened_count_depth_index(
            depth_stats_by_depth=depth_stats_by_depth,
            active_depths=active_depths,
        )
    if depth_selection_policy == "inverse_depth":
        return sample_depth_by_inverse_depth(
            active_depths=active_depths,
            random_generator=random_generator,
        )
    raise invalid_linoo_depth_selection_policy_error(depth_selection_policy)


def choose_depth_by_opened_count_depth_index(
    *,
    depth_stats_by_depth: dict[int, LinooDepthStats],
    active_depths: tuple[int, ...],
) -> int:
    """Choose the depth with the old opened-count depth index."""
    return min(
        active_depths,
        key=lambda depth: (
            depth_stats_by_depth[depth].opened_count * (depth + 1),
            depth,
        ),
    )


def sample_depth_by_inverse_depth(
    *,
    active_depths: tuple[int, ...],
    random_generator: Random,
) -> int:
    """Sample one active depth with inverse-depth weighting."""
    if len(active_depths) == 1:
        return active_depths[0]
    total_weight = sum(inverse_depth_weight(depth) for depth in active_depths)
    threshold = random_generator.random() * total_weight
    cumulative_weight = 0.0
    for depth in active_depths:
        cumulative_weight += inverse_depth_weight(depth)
        if threshold < cumulative_weight:
            return depth
    return active_depths[-1]


def inverse_depth_weight(depth: int) -> float:
    """Return the inverse-depth sampling weight for one depth."""
    return 1.0 / float(depth + 1)


def depth_selection_row_sort_key(
    *,
    depth_selection_policy: LinooDepthSelectionPolicy,
    row: LinooDepthSelectionRow,
) -> tuple[bool, float, int]:
    """Return the report-table ordering key for one depth row."""
    if depth_selection_policy == "inverse_depth":
        probability = (
            -row.selection_probability if row.selection_probability is not None else 0.0
        )
        return (row.selection_probability is None, probability, row.depth)
    selection_index = (
        float(row.selection_index) if row.selection_index is not None else 0.0
    )
    return (row.selection_index is None, selection_index, row.depth)
