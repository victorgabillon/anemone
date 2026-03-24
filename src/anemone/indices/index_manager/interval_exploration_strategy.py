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


class UpdateIndexLocalMinChange:
    """Update exploration indices using the interval/local-min-change strategy."""

    needs_parent_state = True

    def update_root_node_index[NodeT: AlgorithmNode[Any]](
        self,
        root_node: NodeT,
        root_node_exploration_index_data: NodeExplorationData[NodeT, Any] | None,
    ) -> None:
        """Initialize root exploration data for the interval strategy."""
        del root_node

        assert isinstance(root_node_exploration_index_data, IntervalExplo)
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

        assert parent_node_state is not None
        parent_state_with_turn = cast("_StateWithTurn", parent_node_state)
        assert isinstance(parent_node_exploration_index_data, IntervalExplo)
        assert isinstance(child_node_exploration_index_data, IntervalExplo)

        if parent_node_exploration_index_data.index is None:
            child_node_exploration_index_data.index = None
            return

        assert parent_node_exploration_index_data.interval is not None
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
        second_best_branch = top_two_branches.second_best_branch()
        assert best_branch is not None
        assert second_best_branch is not None

        best_child = parent_node.branches_children[best_branch]
        comparison_child = parent_node.branches_children[
            second_best_branch
            if child_node == parent_node.branches_children[best_branch]
            else best_branch
        ]
        assert isinstance(best_child, AlgorithmNode)
        assert isinstance(comparison_child, AlgorithmNode)
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
