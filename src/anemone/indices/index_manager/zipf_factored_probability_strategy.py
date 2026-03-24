"""Zipf/factored-probability exploration-index strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.indices.index_manager.exploration_math import (
    zipf_exploration_index,
    zipf_factored_probability,
)
from anemone.indices.node_indices.index_data import (
    NodeExplorationData,
    RecurZipfQuoolExplorationData,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

if TYPE_CHECKING:
    from anemone.trees.tree import Tree


class UpdateIndexZipfFactoredProba:
    """Update exploration indices using the Zipf/factored-probability strategy."""

    needs_parent_state = False

    def update_root_node_index[NodeT: AlgorithmNode[Any]](
        self,
        root_node: NodeT,
        root_node_exploration_index_data: NodeExplorationData[NodeT, Any] | None,
    ) -> None:
        """Initialize root exploration data for the Zipf strategy."""
        del root_node

        assert isinstance(
            root_node_exploration_index_data,
            RecurZipfQuoolExplorationData,
        )
        root_node_exploration_index_data.zipf_factored_proba = 1
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
        """Update one child exploration index under the Zipf strategy."""
        del parent_node, parent_node_state

        assert isinstance(
            parent_node_exploration_index_data,
            RecurZipfQuoolExplorationData,
        )
        assert isinstance(
            child_node_exploration_index_data,
            RecurZipfQuoolExplorationData,
        )

        parent_zipf_factored_proba = (
            parent_node_exploration_index_data.zipf_factored_proba
        )
        assert parent_zipf_factored_proba is not None

        child_zipf_factored_proba = zipf_factored_probability(
            child_rank=child_rank,
            parent_zipf_factored_proba=parent_zipf_factored_proba,
        )
        child_index = zipf_exploration_index(
            zipf_factored_proba_value=child_zipf_factored_proba,
            node_depth=tree.node_depth(child_node),
        )

        if child_node_exploration_index_data.index is None:
            child_node_exploration_index_data.index = child_index
            child_node_exploration_index_data.zipf_factored_proba = (
                child_zipf_factored_proba
            )
            return

        assert child_node_exploration_index_data.zipf_factored_proba is not None
        child_node_exploration_index_data.index = min(
            child_node_exploration_index_data.index,
            child_index,
        )
        child_node_exploration_index_data.zipf_factored_proba = min(
            child_node_exploration_index_data.zipf_factored_proba,
            child_zipf_factored_proba,
        )
