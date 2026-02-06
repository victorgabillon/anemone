"""Define the interface for a tree node used in branch selection.

The `ITreeNode` protocol represents a node in a tree structure for exploring
possible branches. It provides properties and methods for accessing information
about the node, such as its ID, state, depth, child nodes, and parent nodes.

The `ITreeNode` protocol also defines methods for adding a parent node,
generating a dot description for visualization, checking if all branches have
been generated, accessing the available branches, and checking if the state is
terminal.

Note: This is an interface and should not be instantiated directly.
"""

from collections.abc import MutableMapping
from typing import Protocol, Self

from valanga import BranchKey, BranchKeyGeneratorP, State, StateTag


class ITreeNode[StateT: State = State](Protocol):
    """The `ITreeNode` protocol represents a node in a tree structure used for selecting branches."""

    @property
    def id(self) -> int:
        """Get the ID of the node.

        Returns:
            The ID of the node.

        """
        ...

    @property
    def state(self) -> StateT:
        """Get the state of the node.

        Returns:
            The state of the node.

        """
        ...

    @property
    def tree_depth(self) -> int:
        """Get the tree depth of the node.

        Returns:
            The tree depth of the node.

        """
        ...

    @property
    def branches_children(self) -> MutableMapping[BranchKey, Self | None]:
        """Get the child nodes of the node.

        Returns:
            A bidirectional dictionary mapping branches to child nodes.

        """
        ...

    @property
    def parent_nodes(self) -> dict[Self, BranchKey]:
        """Returns the dictionary of parent nodes of the current tree node with associated branch.

        :return: A dictionary of parent nodes of the current tree node with associated branch.
        """
        ...

    def add_parent(self, branch_key: BranchKey, new_parent_node: Self) -> None:
        """Add a parent node to the node.

        Args:
            new_parent_node: The parent node to add.
            branch_key: The branch key that led to the node from the new parent.

        """

    def dot_description(self) -> str:
        """Generate a dot description for visualization.

        Returns:
            A string containing the dot description.

        """
        ...

    @property
    def all_branches_generated(self) -> bool:
        """Check if all branches have been generated.

        Returns:
            True if all branches have been generated, False otherwise.

        """
        ...

    @all_branches_generated.setter
    def all_branches_generated(self, value: bool) -> None:
        """Set the flag indicating that all branches have been generated."""

    @property
    def all_branches_keys(self) -> BranchKeyGeneratorP[BranchKey]:
        """Get the available branch keys of the node.

        Returns:
            A generator for iterating over the branch keys.

        """
        ...

    @property
    def tag(self) -> StateTag:
        """Get the fast tag representation of the node.

        Returns:
            The fast tag representation of the node as a string.

        """
        ...

    def is_over(self) -> bool:
        """Check if the state is terminal.

        Returns:
            True if the state is terminal, False otherwise.

        """
        ...
