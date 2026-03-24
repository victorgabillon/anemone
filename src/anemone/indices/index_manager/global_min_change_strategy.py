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


def _wrong_minmax_path_value_data_error(
    *,
    node_role: str,
    node_id: object,
    actual_data: object | None,
) -> TypeError:
    return TypeError(
        "Global-min-change strategy requires MinMaxPathValue data on "
        f"{node_role} node {node_id}, got {type(actual_data).__name__}."
    )


def _missing_parent_path_value_error(
    *,
    child_node_id: object,
    parent_node_id: object,
    field_name: str,
) -> RuntimeError:
    return RuntimeError(
        "Cannot update global-min-change indices for child node "
        f"{child_node_id}: parent node {parent_node_id} has no {field_name} yet."
    )


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

        if not isinstance(root_node_exploration_index_data, MinMaxPathValue):
            raise _wrong_minmax_path_value_data_error(
                node_role="root",
                node_id=root_node.id,
                actual_data=root_node_exploration_index_data,
            )
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
        del child_rank, parent_node_state, tree

        if not isinstance(parent_node_exploration_index_data, MinMaxPathValue):
            raise _wrong_minmax_path_value_data_error(
                node_role="parent",
                node_id=parent_node.id,
                actual_data=parent_node_exploration_index_data,
            )
        if parent_node_exploration_index_data.min_path_value is None:
            raise _missing_parent_path_value_error(
                child_node_id=child_node.id,
                parent_node_id=parent_node.id,
                field_name="min_path_value",
            )
        if parent_node_exploration_index_data.max_path_value is None:
            raise _missing_parent_path_value_error(
                child_node_id=child_node.id,
                parent_node_id=parent_node.id,
                field_name="max_path_value",
            )
        if not isinstance(child_node_exploration_index_data, MinMaxPathValue):
            raise _wrong_minmax_path_value_data_error(
                node_role="child",
                node_id=child_node.id,
                actual_data=child_node_exploration_index_data,
            )

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

        assert child_node_exploration_index_data.max_path_value is not None, (
            "Global-min-change exploration data with an existing index must also "
            "store max_path_value."
        )
        assert child_node_exploration_index_data.min_path_value is not None, (
            "Global-min-change exploration data with an existing index must also "
            "store min_path_value."
        )
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
