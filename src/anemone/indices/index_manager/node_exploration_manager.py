"""Public coordination surface for exploration-index updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from anemone.indices.index_manager.global_min_change_strategy import (
    UpdateIndexGlobalMinChange,
)
from anemone.indices.index_manager.interval_exploration_strategy import (
    UpdateIndexLocalMinChange,
)
from anemone.indices.index_manager.zipf_factored_probability_strategy import (
    UpdateIndexZipfFactoredProba,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.basics import TreeDepth
    from anemone.indices.node_indices.index_data import NodeExplorationData
    from anemone.node_evaluation.common.branch_ordering import (
        DecisionOrderedEvaluation,
    )
    from anemone.trees.descendants import RangedDescendants
    from anemone.trees.tree import Tree


class NodeExplorationIndexManager(Protocol):
    """Strategy surface for recomputing exploration indices across a tree."""

    needs_parent_state: bool

    def update_root_node_index[NodeT: AlgorithmNode[Any]](
        self,
        root_node: NodeT,
        root_node_exploration_index_data: NodeExplorationData[NodeT, Any] | None,
    ) -> None:
        """Initialize the root node's exploration-index data."""
        ...

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
        """Update one child node's exploration-index data from one parent path."""
        ...


class NullNodeExplorationIndexManager(NodeExplorationIndexManager):
    """No-op exploration-index strategy used when no index is configured."""

    needs_parent_state = False

    def update_root_node_index[NodeT: AlgorithmNode[Any]](
        self,
        root_node: NodeT,
        root_node_exploration_index_data: NodeExplorationData[NodeT, Any] | None,
    ) -> None:
        """Leave root exploration-index data unchanged."""
        del root_node, root_node_exploration_index_data

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
        """Reject child-index updates when the null strategy was bypassed."""
        del (
            child_node,
            parent_node,
            parent_node_exploration_index_data,
            child_node_exploration_index_data,
            parent_node_state,
            tree,
            child_rank,
        )
        raise NotImplementedError("should not be raised")


def update_all_indices[NodeT: AlgorithmNode[Any]](
    tree: Tree[NodeT],
    index_manager: NodeExplorationIndexManager,
) -> None:
    """Recompute exploration indices across the whole tree using one strategy."""
    if isinstance(index_manager, NullNodeExplorationIndexManager):
        return

    tree_nodes: RangedDescendants[NodeT] = tree.descendants
    index_manager.update_root_node_index(
        root_node=tree.root_node,
        root_node_exploration_index_data=tree.root_node.exploration_index_data,
    )

    tree_depth: TreeDepth
    for tree_depth in tree_nodes:
        for parent_node in tree_nodes[tree_depth].values():
            tree_eval = cast("DecisionOrderedEvaluation", parent_node.tree_evaluation)
            ordered_branches: list[BranchKey] = tree_eval.decision_ordered_branches()
            for child_rank, branch in enumerate(ordered_branches):
                child_node = parent_node.branches_children[branch]
                if child_node is None:
                    continue

                parent_state = (
                    parent_node.state if index_manager.needs_parent_state else None
                )
                index_manager.update_node_indices(
                    child_node=child_node,
                    parent_node=parent_node,
                    parent_node_exploration_index_data=parent_node.exploration_index_data,
                    child_node_exploration_index_data=child_node.exploration_index_data,
                    parent_node_state=parent_state,
                    tree=tree,
                    child_rank=child_rank,
                )


def print_all_indices[NodeT: AlgorithmNode[Any]](
    tree: Tree[NodeT],
) -> None:
    """Print the exploration index currently stored on each node."""
    tree_nodes: RangedDescendants[NodeT] = tree.descendants

    tree_depth: TreeDepth
    for tree_depth in tree_nodes:
        for parent_node in tree_nodes[tree_depth].values():
            if parent_node.exploration_index_data is not None:
                print(
                    "parent_node",
                    parent_node.id,
                    parent_node.exploration_index_data.index,
                )


__all__ = [
    "NodeExplorationIndexManager",
    "NullNodeExplorationIndexManager",
    "UpdateIndexGlobalMinChange",
    "UpdateIndexLocalMinChange",
    "UpdateIndexZipfFactoredProba",
    "print_all_indices",
    "update_all_indices",
]
