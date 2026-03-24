"""Interval-based exploration-index strategy."""

from __future__ import annotations

from math import inf
from typing import TYPE_CHECKING, Any, Protocol, cast

from valanga import Color, State

from anemone.indices.index_manager.exploration_math import node_score
from anemone.indices.node_indices.index_data import (
    IntervalExplo,
    NodeExplorationData,
)
from anemone.node_evaluation.common.branch_ordering import (
    require_second_best_branch_aware,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.utils.small_tools import (
    Interval,
    distance_number_to_interval,
    intersect_intervals,
)

if TYPE_CHECKING:
    from anemone.trees.tree import Tree


class _StateWithTurn(State, Protocol):
    turn: Color


def _wrong_interval_data_error(
    *,
    node_role: str,
    node_id: object,
    actual_data: object | None,
) -> TypeError:
    return TypeError(
        "Interval exploration strategy requires IntervalExplo data on "
        f"{node_role} node {node_id}, got {type(actual_data).__name__}."
    )


def _missing_parent_state_error(
    *,
    child_node_id: object,
    parent_node_id: object,
) -> RuntimeError:
    return RuntimeError(
        "Cannot update interval exploration indices for child node "
        f"{child_node_id}: parent node {parent_node_id} state was not provided."
    )


def _missing_parent_interval_error(
    *,
    child_node_id: object,
    parent_node_id: object,
) -> RuntimeError:
    return RuntimeError(
        "Cannot update interval exploration indices for child node "
        f"{child_node_id}: parent node {parent_node_id} has an initialized "
        "index but no interval."
    )


def _missing_best_branch_error(
    *,
    child_node_id: object,
    parent_node_id: object,
) -> RuntimeError:
    return RuntimeError(
        "Cannot compute interval exploration update for child node "
        f"{child_node_id}: parent node {parent_node_id} has no best branch yet."
    )


def _missing_best_child_error(
    *,
    parent_node_id: object,
    best_branch: object,
) -> RuntimeError:
    return RuntimeError(
        "Cannot compute interval exploration update: "
        f"parent node {parent_node_id} selected best branch {best_branch!r} "
        "but no child is linked to it."
    )


def _missing_comparison_child_error(
    *,
    parent_node_id: object,
    best_branch: object,
    second_best_branch: object,
) -> RuntimeError:
    return RuntimeError(
        "Cannot compute interval exploration update: "
        f"parent node {parent_node_id} has no comparison child for branches "
        f"{best_branch!r}/{second_best_branch!r}."
    )


class UpdateIndexLocalMinChange:
    """Update exploration indices using the interval/local-min-change strategy."""

    needs_parent_state = True

    def update_root_node_index[NodeT: AlgorithmNode[Any]](
        self,
        root_node: NodeT,
        root_node_exploration_index_data: NodeExplorationData[NodeT, Any] | None,
    ) -> None:
        """Initialize root exploration data for the interval strategy."""
        if not isinstance(root_node_exploration_index_data, IntervalExplo):
            raise _wrong_interval_data_error(
                node_role="root",
                node_id=root_node.id,
                actual_data=root_node_exploration_index_data,
            )
        root_node_exploration_index_data.index = 0
        root_node_exploration_index_data.interval = Interval(
            min_value=-inf,
            max_value=inf,
        )

    def update_node_indices[NodeT: AlgorithmNode[Any]](
        self,
        child_node: NodeT,
        parent_node: NodeT,
        parent_node_exploration_index_data: NodeExplorationData[NodeT, Any] | None,
        child_node_exploration_index_data: NodeExplorationData[NodeT, Any] | None,
        parent_node_state: object | None,
        tree: Tree[NodeT],
        child_rank: int,
    ) -> None:
        """Update one child exploration index under the interval strategy."""
        del child_rank, tree

        if parent_node_state is None:
            raise _missing_parent_state_error(
                child_node_id=child_node.id,
                parent_node_id=parent_node.id,
            )
        parent_state_with_turn = cast("_StateWithTurn", parent_node_state)
        if not isinstance(parent_node_exploration_index_data, IntervalExplo):
            raise _wrong_interval_data_error(
                node_role="parent",
                node_id=parent_node.id,
                actual_data=parent_node_exploration_index_data,
            )
        if not isinstance(child_node_exploration_index_data, IntervalExplo):
            raise _wrong_interval_data_error(
                node_role="child",
                node_id=child_node.id,
                actual_data=child_node_exploration_index_data,
            )

        if parent_node_exploration_index_data.index is None:
            child_node_exploration_index_data.index = None
            return

        if parent_node_exploration_index_data.interval is None:
            raise _missing_parent_interval_error(
                child_node_id=child_node.id,
                parent_node_id=parent_node.id,
            )
        inter_level_interval: Interval | None
        local_index: float | None
        if len(parent_node.branches_children) == 1:
            local_index = parent_node_exploration_index_data.index
            inter_level_interval = parent_node_exploration_index_data.interval
        else:
            inter_level_interval, local_index = self._interval_and_index_from_parent(
                child_node=child_node,
                parent_node=parent_node,
                parent_interval=parent_node_exploration_index_data.interval,
                parent_turn=parent_state_with_turn.turn,
            )

        self._update_child_interval_data(
            child_node_exploration_index_data=child_node_exploration_index_data,
            local_index=local_index,
            inter_level_interval=inter_level_interval,
        )

    def _interval_and_index_from_parent[NodeT: AlgorithmNode[Any]](
        self,
        *,
        child_node: NodeT,
        parent_node: NodeT,
        parent_interval: Interval,
        parent_turn: Color,
    ) -> tuple[Interval | None, float | None]:
        """Return the child interval and local index induced by one parent node."""
        best_child, comparison_child = self._best_and_comparison_child(
            child_node=child_node,
            parent_node=parent_node,
        )

        local_interval = self._local_interval_for_child(
            child_node=child_node,
            best_child=best_child,
            comparison_child=comparison_child,
            parent_turn=parent_turn,
        )
        inter_level_interval = intersect_intervals(local_interval, parent_interval)
        if inter_level_interval is None:
            return None, None

        local_index = distance_number_to_interval(
            value=node_score(child_node),
            interval=inter_level_interval,
        )
        return inter_level_interval, local_index

    def _best_and_comparison_child[NodeT: AlgorithmNode[Any]](
        self,
        *,
        child_node: NodeT,
        parent_node: NodeT,
    ) -> tuple[AlgorithmNode[Any], AlgorithmNode[Any]]:
        """Return the best child plus the branch that constrains ``child_node``."""
        top_two_branches = require_second_best_branch_aware(parent_node.tree_evaluation)
        best_branch = top_two_branches.best_branch()
        if best_branch is None:
            raise _missing_best_branch_error(
                child_node_id=child_node.id,
                parent_node_id=parent_node.id,
            )
        second_best_branch = top_two_branches.second_best_branch()

        best_child = parent_node.branches_children[best_branch]
        if best_child is None:
            raise _missing_best_child_error(
                parent_node_id=parent_node.id,
                best_branch=best_branch,
            )
        comparison_child = parent_node.branches_children[
            second_best_branch
            if child_node == parent_node.branches_children[best_branch]
            else best_branch
        ]
        if comparison_child is None:
            raise _missing_comparison_child_error(
                parent_node_id=parent_node.id,
                best_branch=best_branch,
                second_best_branch=second_best_branch,
            )
        assert isinstance(best_child, AlgorithmNode), (
            "Interval exploration strategy expects AlgorithmNode children for "
            "best-branch comparisons."
        )
        assert isinstance(comparison_child, AlgorithmNode), (
            "Interval exploration strategy expects AlgorithmNode children for "
            "best-branch comparisons."
        )
        return best_child, comparison_child

    def _local_interval_for_child(
        self,
        *,
        child_node: AlgorithmNode[Any],
        best_child: AlgorithmNode[Any],
        comparison_child: AlgorithmNode[Any],
        parent_turn: Color,
    ) -> Interval:
        """Return the local interval induced by one parent's ordered children."""
        del child_node, best_child

        if parent_turn == Color.WHITE:
            return Interval(min_value=node_score(comparison_child), max_value=inf)
        return Interval(min_value=-inf, max_value=node_score(comparison_child))

    def _update_child_interval_data(
        self,
        *,
        child_node_exploration_index_data: IntervalExplo[Any, Any],
        local_index: float | None,
        inter_level_interval: Interval | None,
    ) -> None:
        """Store interval-strategy updates while preserving the current min index."""
        if child_node_exploration_index_data.index is None:
            child_node_exploration_index_data.index = local_index
            child_node_exploration_index_data.interval = inter_level_interval
            return

        if local_index is None:
            return

        if local_index < child_node_exploration_index_data.index:
            child_node_exploration_index_data.interval = inter_level_interval
        child_node_exploration_index_data.index = min(
            child_node_exploration_index_data.index,
            local_index,
        )
