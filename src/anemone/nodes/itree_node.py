"""Structural protocol shared by tree-node wrappers.

`ITreeNode` defines the navigation and branch-opening surface that generic tree
helpers are allowed to depend on. It intentionally stays structural: search
runtime details such as evaluations, exploration indices, and PV bookkeeping
belong on higher-level wrappers like `AlgorithmNode`, not on this protocol.
"""

from collections.abc import MutableMapping
from typing import Protocol, Self

from valanga import BranchKey, State, StateTag


class ITreeNode[StateT: State = State](Protocol):
    """Structural/navigation protocol for nodes stored in a tree."""

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
    def parent_nodes(self) -> dict[Self, set[BranchKey]]:
        """Return the incoming parent-edge mapping for this node.

        Each key is a parent node. Each value is the set of distinct branch keys
        through which that parent reaches this node.
        """
        ...

    def add_parent(self, branch_key: BranchKey, new_parent_node: Self) -> None:
        """Add a parent node to the node.

        Args:
            new_parent_node: The parent node to add.
            branch_key: The branch key that led to the node from the new parent.

        """

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
    def non_opened_branches(self) -> set[BranchKey]:
        """Return structural branches that exist conceptually but are not opened yet."""
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
