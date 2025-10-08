"""
This module defines the AlgorithmNode class, which is a generic node used by the tree and value algorithm.
It wraps tree nodes with values, minimax computation, and exploration tools.
"""

from anemone.basics import (
    BranchKey,
    BranchKeyGeneratorP,
    ContentRepresentation,
    ContentTag,
    NodeContent,
)
from anemone.indices.node_indices import NodeExplorationData
from anemone.nodes.algorithm_node.node_minmax_evaluation import NodeMinmaxEvaluation
from anemone.nodes.itree_node import ITreeNode
from anemone.nodes.tree_node import TreeNode


class AlgorithmNode:
    """
    The generic Node used by the tree and value algorithm.
    It wraps tree nodes with values, minimax computation and exploration tools
    """

    tree_node: TreeNode  # the reference to the tree node that is wrapped
    minmax_evaluation: NodeMinmaxEvaluation  # the object computing the value
    exploration_index_data: (
        NodeExplorationData | None
    )  # the object storing the information to help the algorithm decide the next nodes to explore
    content_representation: ContentRepresentation | None  # the board representation

    def __init__(
        self,
        tree_node: TreeNode,
        minmax_evaluation: NodeMinmaxEvaluation,
        exploration_index_data: NodeExplorationData | None,
        content_representation: ContentRepresentation | None,
    ) -> None:
        """
        Initializes an AlgorithmNode object.

        Args:
            tree_node (TreeNode): The tree node that is wrapped.
            minmax_evaluation (NodeMinmaxEvaluation): The object computing the value.
            exploration_index_data (NodeExplorationData | None): The object storing the information to help the algorithm decide the next nodes to explore.
            content_representation (ContentRepresentation | None): The board representation.
        """
        self.tree_node = tree_node
        self.minmax_evaluation = minmax_evaluation
        self.exploration_index_data = exploration_index_data
        self.content_representation = content_representation

    @property
    def id(self) -> int:
        """
        Returns the ID of the node.

        Returns:
            int: The ID of the node.
        """
        return self.tree_node.id

    @property
    def tree_depth(self) -> int:
        """
        Returns the tree depth.

        Returns:
            int: The tree depth.
        """
        return self.tree_node.tree_depth_

    @property
    def tag(self) -> ContentTag:
        """
        Returns the fast representation of the node.

        Returns:
            str: The fast representation of the node.
        """
        return self.tree_node.tag

    @property
    def branches_children(self) -> dict[BranchKey, ITreeNode | None]:
        """
        Returns the bidirectional dictionary of moves and their corresponding child nodes.

        Returns:
            dict[IMove, ITreeNode | None]: The bidirectional dictionary of moves and their corresponding child nodes.
        """
        return self.tree_node.branches_children

    @property
    def parent_nodes(self) -> dict[ITreeNode, BranchKey]:
        """
        Returns the dictionary of parent nodes of the current tree node with associated move.

        :return: A dictionary of parent nodes of the current tree node with associated move.
        """
        return self.tree_node.parent_nodes

    @property
    def content(self) -> NodeContent:
        """
        Returns the content associated with this tree node.

        Returns:
            ContentWithTag: The content associated with this tree node.
        """
        return self.tree_node.content

    def is_over(self) -> bool:
        """
        Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.minmax_evaluation.is_over()

    def add_parent(self, branch_key: BranchKey, new_parent_node: ITreeNode) -> None:
        """
        Adds a parent node.

        Args:
            branch_key (BranchKey): The branch key associated with the move that led to the node from the new_parent_node.
            new_parent_node (ITreeNode): The new parent node to add.
        """
        self.tree_node.add_parent(
            branch_key=branch_key, new_parent_node=new_parent_node
        )

    @property
    def all_branches_keys(self) -> BranchKeyGeneratorP:
        """
        Returns a generator that yields the branch keys for the current board state.

        Returns:
            BranchKeyGenerator: A generator that yields the branch keys.
        """
        return self.tree_node.content_.branch_keys

    @property
    def all_branches_generated(self) -> bool:
        """
        Returns True if all branches have been generated, False otherwise.

        Returns:
            bool: True if all branches have been generated, False otherwise.
        """
        return self.tree_node.all_branches_generated

    @all_branches_generated.setter
    def all_branches_generated(self, value: bool) -> None:
        """
        Sets the flag indicating if all branches have been generated.

        Args:
            value (bool): The value to set.
        """
        self.tree_node.all_branches_generated = value

    @property
    def non_opened_branches(self) -> set[BranchKey]:
        """
        Returns the set of non-opened branches.

        Returns:
            set[BranchKey]: The set of non-opened branches.
        """
        return self.tree_node.non_opened_branches

    def dot_description(self) -> str:
        """
        Returns the dot description of the node.

        Returns:
            str: The dot description of the node.
        """
        exploration_description: str = (
            self.exploration_index_data.dot_description()
            if self.exploration_index_data is not None
            else ""
        )

        return f"{self.tree_node.dot_description()}\n{self.minmax_evaluation.dot_description()}\n{exploration_description}"

    def __str__(self) -> str:
        return f"{self.__class__} id :{self.tree_node.id}"
