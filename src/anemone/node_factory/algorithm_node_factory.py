"""Provide the algorithm node factory."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from valanga import (
    BranchKey,
    ContentRepresentation,
    RepresentationFactory,
    State,
    StateModifications,
)
from valanga.evaluator_types import EvaluatorInput

from anemone.basics import TreeDepth
from anemone.indices import node_indices
from anemone.node_evaluation.tree.factory import (
    NodeTreeEvaluationFactory,
)
from anemone.node_factory.base import TreeNodeFactory
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from anemone.nodes.state_handles import MaterializedStateHandle, StateHandle
from anemone.nodes.tree_node import TreeNode

if TYPE_CHECKING:
    from anemone.node_evaluation.tree.node_tree_evaluation import (
        NodeTreeEvaluation,
    )


@dataclass
class AlgorithmNodeFactory[StateT: State = State]:
    """Factory for building AlgorithmNode instances."""

    tree_node_factory: TreeNodeFactory[AlgorithmNode[StateT], StateT]
    state_representation_factory: (
        RepresentationFactory[StateT, EvaluatorInput, StateModifications] | None
    )
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT]
    exploration_index_data_create: node_indices.ExplorationIndexDataFactory[
        AlgorithmNode[StateT], StateT
    ]

    def create_from_tree_node(
        self,
        tree_node: TreeNode[AlgorithmNode[StateT], StateT],
        parent_node: AlgorithmNode[StateT] | None,
        modifications: StateModifications | None,
        build_state_representation: bool = True,
    ) -> AlgorithmNode[StateT]:
        """Build an AlgorithmNode from an existing TreeNode."""
        tree_evaluation: NodeTreeEvaluation[StateT] = (
            self.node_tree_evaluation_factory.create(
                tree_node=tree_node,
            )
        )

        exploration_index_data: (
            node_indices.NodeExplorationData[AlgorithmNode[StateT], StateT] | None
        ) = self.exploration_index_data_create(tree_node)

        state_representation: ContentRepresentation[StateT, EvaluatorInput] | None = (
            None
        )
        if self.state_representation_factory is not None and build_state_representation:
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
        state_handle: StateHandle[StateT],
        tree_depth: TreeDepth,
        count: int,
        parent_node: AlgorithmNode[StateT] | None,
        branch_from_parent: BranchKey | None,
        modifications: StateModifications | None,
        build_state_representation: bool = True,
    ) -> AlgorithmNode[StateT]:
        """Create an AlgorithmNode object.

        Args:
            branch_from_parent: The branch key leading from the parent node.
            state_handle: The explicit state handle for the node.
            tree_depth: The depth of the node in the tree.
            count: The node identifier.
            parent_node: The parent node object.
            modifications: The state modifications object.
            build_state_representation: Whether to eagerly build the optional
                evaluator-side state representation.

        Returns:
            The created AlgorithmNode.

        """
        tree_node: TreeNode[AlgorithmNode[StateT], StateT] = (
            self.tree_node_factory.create(
                state_handle=state_handle,
                tree_depth=tree_depth,
                count=count,
                branch_from_parent=branch_from_parent,
                parent_node=parent_node,
            )
        )

        return self.create_from_tree_node(
            tree_node=tree_node,
            parent_node=parent_node,
            modifications=modifications,
            build_state_representation=build_state_representation,
        )

    def create_from_state(
        self,
        state: StateT,
        tree_depth: TreeDepth,
        count: int,
        parent_node: AlgorithmNode[StateT] | None,
        branch_from_parent: BranchKey | None,
        modifications: StateModifications | None,
        build_state_representation: bool = True,
    ) -> AlgorithmNode[StateT]:
        """Convenience wrapper that materializes a handle from a concrete state."""
        return self.create(
            state_handle=MaterializedStateHandle(state_=state),
            tree_depth=tree_depth,
            count=count,
            parent_node=parent_node,
            branch_from_parent=branch_from_parent,
            modifications=modifications,
            build_state_representation=build_state_representation,
        )
