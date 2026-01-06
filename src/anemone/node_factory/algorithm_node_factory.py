""" "
AlgorithmNodeFactory
"""

from dataclasses import dataclass
from valanga import (
    BranchKey,
    ContentRepresentation,
    RepresentationFactory,
    State,
    StateModifications,
)

import anemone.indices.node_indices as node_indices
from anemone.basics import TreeDepth
from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation import NodeTreeEvaluation
from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation_factory import NodeTreeEvaluationFactory
from anemone.node_factory.base import TreeNodeFactory
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from anemone.nodes.tree_node import TreeNode


@dataclass
class AlgorithmNodeFactory[TState: State = State]:
    """
    The classe creating Algorithm Nodes
    """

    tree_node_factory: TreeNodeFactory[AlgorithmNode[TState],TState]
    state_representation_factory: RepresentationFactory | None
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[TState]
    exploration_index_data_create: node_indices.ExplorationIndexDataFactory[AlgorithmNode[TState],TState]  



    def create_from_tree_node(
        self,
        tree_node: TreeNode[AlgorithmNode[TState], TState],
        parent_node: AlgorithmNode[TState] | None,
        modifications: StateModifications | None,
    ) -> AlgorithmNode[TState]:
        

        tree_evaluation: NodeTreeEvaluation[TState] = self.node_tree_evaluation_factory.create(
            tree_node=tree_node,
        )

        exploration_index_data: (
            node_indices.NodeExplorationData[AlgorithmNode[TState], TState] | None
        ) = self.exploration_index_data_create(tree_node)

        state_representation: ContentRepresentation | None = None
        if self.state_representation_factory is not None:
            if parent_node is not None:
                parent_node_representation = parent_node.state_representation
            else:
                parent_node_representation = None

            state_representation = (
                self.state_representation_factory.create_from_transition(
                    state=tree_node.state,
                    previous_state_representation=parent_node_representation,
                    modifications=modifications,
                )
            )

        return AlgorithmNode(
            tree_node=tree_node,
            tree_evaluation=tree_evaluation,
            exploration_index_data=exploration_index_data,
            state_representation=state_representation,
        )

    def create(
        self,
        state: TState,
        tree_depth: TreeDepth,
        count: int,
        parent_node: AlgorithmNode[TState] | None,
        branch_from_parent: BranchKey | None,
        modifications: StateModifications | None,
    ) -> AlgorithmNode[TState]:
        """
        Creates an AlgorithmNode object.

        Args:
            branch_from_parent (BranchKey | None): the move that led to the node from the parent node
            state: The state object.
            tree_depth: The tree depth.
            count: The count.
            parent_node: The parent node object.
            modifications: The board modifications object.

        Returns:
            An AlgorithmNode object.

        """
        tree_node: TreeNode[AlgorithmNode[TState], TState] = self.tree_node_factory.create(
            state=state,
            tree_depth=tree_depth,
            count=count,
            branch_from_parent=branch_from_parent,
            parent_node=parent_node,
        )

        return self.create_from_tree_node(
            tree_node=tree_node,
            parent_node=parent_node,
            modifications=modifications,
        )
