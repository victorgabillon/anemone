"""Global-min-change exploration-index strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.indices.index_manager.exploration_math import (
    global_min_change_metrics,
    node_score,
)
from anemone.indices.node_indices.index_data import (
    MinMaxPathValue,
    NodeExplorationData,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

if TYPE_CHECKING:
    from anemone.trees.tree import Tree


class UpdateIndexGlobalMinChange:
    """Update exploration indices using the global minimum-change strategy."""

    needs_parent_state = False

    def update_root_node_index[NodeT: AlgorithmNode[Any]](
        self,
        root_node: NodeT,
        root_node_exploration_index_data: NodeExplorationData[NodeT, Any] | None,
    ) -> None:
        """Initialize root exploration data for the global-min-change strategy."""
        root_value = node_score(root_node)

        assert isinstance(root_node_exploration_index_data, MinMaxPathValue)
        root_node_exploration_index_data.min_path_value = root_value
        root_node_exploration_index_data.max_path_value = root_value
        root_node_exploration_index_data.index = 0

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
        """Update one child exploration index under the global-min-change rule."""
        del child_rank, parent_node, parent_node_state, tree

        assert isinstance(parent_node_exploration_index_data, MinMaxPathValue)
        assert parent_node_exploration_index_data.min_path_value is not None
        assert parent_node_exploration_index_data.max_path_value is not None
        assert isinstance(child_node_exploration_index_data, MinMaxPathValue)

        child_min_path_value, child_max_path_value, child_index = (
            global_min_change_metrics(
                child_score=node_score(child_node),
                parent_min_path_value=parent_node_exploration_index_data.min_path_value,
                parent_max_path_value=parent_node_exploration_index_data.max_path_value,
            )
        )

        if child_node_exploration_index_data.index is None:
            child_node_exploration_index_data.index = child_index
            child_node_exploration_index_data.max_path_value = child_max_path_value
            child_node_exploration_index_data.min_path_value = child_min_path_value
            return

        assert child_node_exploration_index_data.max_path_value is not None
        assert child_node_exploration_index_data.min_path_value is not None
        child_node_exploration_index_data.index = min(
            child_node_exploration_index_data.index,
            child_index,
        )
        child_node_exploration_index_data.max_path_value = min(
            child_max_path_value,
            child_node_exploration_index_data.max_path_value,
        )
        child_node_exploration_index_data.min_path_value = max(
            child_min_path_value,
            child_node_exploration_index_data.min_path_value,
        )
