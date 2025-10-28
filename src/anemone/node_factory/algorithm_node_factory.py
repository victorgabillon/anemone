""" "
AlgorithmNodeFactory
"""

from dataclasses import dataclass
from typing import Any

from valanga import (
    BranchKey,
    ContentRepresentation,
    RepresentationFactory,
    State,
    StateModifications,
)

import anemone.indices.node_indices as node_indices
from anemone.basics import TreeDepth
from anemone.node_factory.base import Base
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from anemone.nodes.algorithm_node.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.nodes.tree_node import TreeNode


@dataclass
class AlgorithmNodeFactory:
    """
    The classe creating Algorithm Nodes
    """

    tree_node_factory: Base[Any]
    content_representation_factory: RepresentationFactory | None
    exploration_index_data_create: node_indices.ExplorationIndexDataFactory[
        Any
    ]  # Use Any to avoid protocol constraints

    def create(
        self,
        state: State,
        tree_depth: TreeDepth,
        count: int,
        parent_node: AlgorithmNode | None,
        branch_from_parent: BranchKey | None,
        modifications: StateModifications | None,
    ) -> AlgorithmNode:
        """
        Creates an AlgorithmNode object.

        Args:
            branch_from_parent (BranchKey | None): the move that led to the node from the parent node
            content: The content object.
            tree_depth: The tree depth.
            count: The count.
            parent_node: The parent node object.
            modifications: The board modifications object.

        Returns:
            An AlgorithmNode object.

        """
        tree_node: TreeNode[AlgorithmNode] = self.tree_node_factory.create(
            state=state,
            tree_depth=tree_depth,
            count=count,
            branch_from_parent=branch_from_parent,
            parent_node=parent_node,
        )
        minmax_evaluation: NodeMinmaxEvaluation[AlgorithmNode] = NodeMinmaxEvaluation(
            tree_node=tree_node
        )

        exploration_index_data: (
            node_indices.NodeExplorationData[AlgorithmNode] | None
        ) = self.exploration_index_data_create(tree_node)

        content_representation: ContentRepresentation | None = None
        if self.content_representation_factory is not None:
            if parent_node is not None:
                parent_node_representation = parent_node.content_representation
            else:
                parent_node_representation = None

            content_representation = (
                self.content_representation_factory.create_from_transition(
                    state=tree_node.content,
                    previous_state_representation=parent_node_representation,
                    modifications=modifications,
                )
            )

        return AlgorithmNode(
            tree_node=tree_node,
            minmax_evaluation=minmax_evaluation,
            exploration_index_data=exploration_index_data,
            content_representation=content_representation,
        )
