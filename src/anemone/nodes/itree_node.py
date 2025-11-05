"""
This module defines the interface for a tree node in a chess move selector.

The `ITreeNode` protocol represents a node in a tree structure used for selecting chess moves.
It provides properties and methods for accessing information about the node, such as its ID,
the chess board state, the half move count, the child nodes, and the parent nodes.

The `ITreeNode` protocol also defines methods for adding a parent node, generating a dot description
for visualization, checking if all legal moves have been generated, accessing the legal moves,
and checking if the game is over.

Note: This is an interface and should not be instantiated directly.
"""

from typing import Any, MutableMapping, Protocol

from valanga import BranchKey, BranchKeyGeneratorP, State, StateTag


class ITreeNode[OtherNode: "ITreeNode" = Any](Protocol):
    """
    The `ITreeNode` protocol represents a node in a tree structure used for selecting chess moves.
    """

    @property
    def id(self) -> int:
        """
        Get the ID of the node.

        Returns:
            The ID of the node.
        """
        ...

    @property
    def state(self) -> State:
        """
        Get the chess board state of the node.

        Returns:
            The chess board state of the node.
        """
        ...

    @property
    def tree_depth(self) -> int:
        """
        Get the tree depth of the node.

        Returns:
            The tree depth of the node.
        """
        ...

    @property
    def branches_children(self) -> MutableMapping[BranchKey, OtherNode | None]:
        """
        Get the child nodes of the node.

        Returns:
            A bidirectional dictionary mapping branches to child nodes.
        """
        ...

    @property
    def parent_nodes(self) -> dict["ITreeNode", BranchKey]:
        """
        Returns the dictionary of parent nodes of the current tree node with associated move.

        :return: A dictionary of parent nodes of the current tree node with associated move.
        """
        ...

    def add_parent(self, branch_key: BranchKey, new_parent_node: "ITreeNode") -> None:
        """
        Add a parent node to the node.

        Args:
            new_parent_node: The parent node to add.
            move (chess.Move): the move that led to the node from the new_parent_node

        """

    def dot_description(self) -> str:
        """
        Generate a dot description for visualization.

        Returns:
            A string containing the dot description.
        """
        ...

    @property
    def all_branches_generated(self) -> bool:
        """
        Check if all branches have been generated.

        Returns:
            True if all branches have been generated, False otherwise.
        """
        ...

    @all_branches_generated.setter
    def all_branches_generated(self, value: bool) -> None:
        """
        Set the flag indicating that all branches have been generated.
        """

    @property
    def all_branches_keys(self) -> BranchKeyGeneratorP:
        """
        Get the legal moves of the node.

        Returns:
            A generator for iterating over the legal moves.
        """
        ...

    @property
    def tag(self) -> StateTag:
        """
        Get the fast representation of the node.

        Returns:
            The fast representation of the node as a string.
        """
        ...

    def is_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            True if the game is over, False otherwise.
        """
        ...
