"""Runtime/search wrapper built on top of a structural ``TreeNode``."""

from collections.abc import MutableMapping
from typing import Self

from valanga import (
    BranchKey,
    ContentRepresentation,
    State,
    StateTag,
)
from valanga.evaluator_types import EvaluatorInput

from anemone.indices.node_indices import NodeExplorationData
from anemone.node_evaluation.tree.node_tree_evaluation import (
    NodeTreeEvaluation,
)
from anemone.nodes.tree_node import TreeNode


class AlgorithmNode[StateT: State = State]:
    """Algorithm-facing wrapper that still exposes the structural ``ITreeNode`` surface.

    ``AlgorithmNode`` owns search/runtime concerns such as tree evaluation,
    exploration-index payloads, and optional evaluator representations while
    delegating structural navigation to the wrapped ``TreeNode``.
    """

    tree_node: TreeNode[Self, StateT]
    # the reference to the tree node that is wrapped pointing to other AlgorithmNodes
    tree_evaluation: NodeTreeEvaluation[StateT]
    exploration_index_data: (
        NodeExplorationData[Self, StateT] | None
    )  # the object storing the information to help the algorithm decide the next nodes to explore
    _state_representation: (
        ContentRepresentation[StateT, EvaluatorInput] | None
    )  # the state representation for evaluation

    @property
    def state_representation(
        self,
    ) -> ContentRepresentation[StateT, EvaluatorInput] | None:
        """Returns the state representation."""
        return self._state_representation

    def __init__(
        self,
        tree_node: TreeNode[Self, StateT],
        tree_evaluation: NodeTreeEvaluation[StateT],
        exploration_index_data: NodeExplorationData[Self, StateT] | None,
        state_representation: ContentRepresentation[StateT, EvaluatorInput] | None,
    ) -> None:
        """Initialize an AlgorithmNode object.

        Args:
            tree_node (TreeNode): The tree node that is wrapped.
            tree_evaluation (NodeTreeEvaluation): The object computing tree-search value and branch decisions.
            exploration_index_data (NodeExplorationData | None): The object storing the information to help the algorithm decide the next nodes to explore.
            state_representation (ContentRepresentation | None): The state representation used for evaluation.

        """
        self.tree_node = tree_node
        self.tree_evaluation = tree_evaluation
        self.exploration_index_data = exploration_index_data
        self._state_representation = state_representation

    @property
    def id(self) -> int:
        """Returns the ID of the node.

        Returns:
            int: The ID of the node.

        """
        return self.tree_node.id

    @property
    def tree_depth(self) -> int:
        """Returns the tree depth.

        Returns:
            int: The tree depth.

        """
        return self.tree_node.tree_depth_

    @property
    def tag(self) -> StateTag:
        """Returns the fast representation of the node.

        Returns:
            str: The fast representation of the node.

        """
        return self.tree_node.tag

    @property
    def branches_children(self) -> MutableMapping[BranchKey, Self | None]:
        """Returns the bidirectional dictionary of branches and their corresponding child nodes.

        Returns:
            dict[BranchKey, ITreeNode | None]: The bidirectional dictionary of branches and their corresponding child nodes.

        """
        return self.tree_node.branches_children

    @property
    def parent_nodes(self) -> dict[Self, set[BranchKey]]:
        """Return the incoming parent-edge mapping for this node."""
        return self.tree_node.parent_nodes

    @property
    def state(self) -> StateT:
        """Returns the state associated with this tree node.

        Returns:
            StateWithTag: The state associated with this tree node.

        """
        return self.tree_node.state

    def is_over(self) -> bool:
        """Return whether this node's own state is terminal.

        Returns:
            bool: True if the node is terminal, False otherwise.

        """
        return self.tree_evaluation.is_terminal()

    def add_parent(self, branch_key: BranchKey, new_parent_node: Self) -> None:
        """Add a parent node.

        Args:
            branch_key (BranchKey): The branch key associated with the branch that led to the node from the new_parent_node.
            new_parent_node (ITreeNode): The new parent node to add.

        """
        self.tree_node.add_parent(
            branch_key=branch_key, new_parent_node=new_parent_node
        )

    @property
    def all_branches_generated(self) -> bool:
        """Returns True if all branches have been generated, False otherwise.

        Returns:
            bool: True if all branches have been generated, False otherwise.

        """
        return self.tree_node.all_branches_generated

    @all_branches_generated.setter
    def all_branches_generated(self, value: bool) -> None:
        """Set the flag indicating if all branches have been generated.

        Args:
            value (bool): The value to set.

        """
        self.tree_node.all_branches_generated = value

    @property
    def non_opened_branches(self) -> set[BranchKey]:
        """Returns the set of non-opened branches.

        Returns:
            set[BranchKey]: The set of non-opened branches.

        """
        return self.tree_node.non_opened_branches

    def __str__(self) -> str:
        """Return a concise string representation of the node."""
        return f"{self.__class__} id :{self.tree_node.id}"
