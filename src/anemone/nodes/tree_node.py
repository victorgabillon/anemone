"""
This module defines the TreeNode class, which represents a node in a tree structure.
"""

from dataclasses import dataclass, field
from typing import Any

from valanga import BranchKey, BranchKeyGeneratorP, State, StateTag

from .itree_node import ITreeNode

# todo replace the any with a defaut value in ITReenode when availble in python; 3.13?


@dataclass(slots=True)
class TreeNode[
    FamilyT: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
]:
    r"""
    The TreeNode class stores information about a specific state, including its depth
    and the parent-child relationships with other nodes.

    Attributes:
        id\_ (int): The number to identify this node for easier debugging.
        tree_depth\_ (int): The depth of the node in the tree.
        state\_ (State): The state associated with the node.
        parent_nodes\_ (dict[ITreeNode, BranchKey]): Parent nodes and the branch keys linking them.
        all_branches_generated (bool): Whether all branches have been generated.
        non_opened_branches (set[BranchKey]): The set of non-opened branches.
        branches_children\_ (dict[BranchKey, ITreeNode | None]): The dictionary mapping branches to child nodes.
        tag (str): The fast tag representation of the state.

    Methods:
        id(): Returns the id of the node.
        state(): Returns the state representation.
        tree_depth(): Returns the depth of the node.
        branches_children(): Returns the dictionary mapping branches to child nodes.
        parent_nodes(): Returns the parent node mapping.
        is_root_node(): Checks if the node is a root node.
        all_branches_keys(): Returns available branch keys.
        add_parent(new_parent_node: ITreeNode): Adds a parent node to the current node.
        is_over(): Checks if the state is terminal.
        print_branches_children(): Prints the branches-children links of the node.
        dot_description(): Returns the dot description of the node.
    """

    # id is a number to identify this node for easier debug
    id_: int

    # the tree depth of this node
    tree_depth_: int

    # the node holds a state.
    state_: StateT

    # the set of parent nodes to this node. Note that a node can have multiple parents!
    parent_nodes_: dict[FamilyT, BranchKey]

    # all_branches_generated is a boolean saying whether all branches have been generated.
    # If true the branches are either opened in which case the corresponding opened node is stored in
    # the dictionary self.branches_children, otherwise it is stored in self.non_opened_branches
    all_branches_generated: bool = False

    @staticmethod
    def _empty_non_opened_branches() -> set[BranchKey]:
        """Return a new empty set for non-opened branches."""
        return set()

    @staticmethod
    def _empty_branches_children() -> dict[BranchKey, FamilyT | None]:
        """Return a new empty mapping for branch children."""
        return {}

    non_opened_branches: set[BranchKey] = field(
        default_factory=_empty_non_opened_branches
    )

    # dictionary mapping moves to children nodes. Node is set to None if not created
    branches_children_: dict[BranchKey, FamilyT | None] = field(
        default_factory=_empty_branches_children
    )

    @property
    def tag(self) -> StateTag:
        """Returns the fast tag representation of the state.

        Returns:
            StateTag: The fast tag representation of the state.
        """
        return self.state_.tag

    @property
    def id(self) -> int:
        """
        Returns the ID of the tree node.

        Returns:
            int: The ID of the tree node.
        """
        return self.id_

    @property
    def state(self) -> StateT:
        """
        Returns the state associated with this tree node.

        Returns:
            State: The state associated with this tree node.
        """
        return self.state_

    @property
    def tree_depth(self) -> int:
        """
        Returns the tree depth of this node.

        Returns:
            int: The tree depth of this node.
        """
        return self.tree_depth_

    @property
    def branches_children(self) -> dict[BranchKey, FamilyT | None]:
        """
        Returns a bidirectional dictionary containing the children nodes of the current tree node,
        along with the corresponding branches that lead to each child node.

        Returns:
            dict[BranchKey, ITreeNode | None]: A bidirectional dictionary mapping branches to
            the corresponding child nodes. If a branch does not have a corresponding child node, it is
            mapped to None.
        """
        return self.branches_children_

    @property
    def parent_nodes(self) -> dict[FamilyT, BranchKey]:
        """
        Returns the dictionary of parent nodes of the current tree node with associated branch.

        :return: A dictionary of parent nodes of the current tree node with associated branch.
        """
        return self.parent_nodes_

    def is_root_node(self) -> bool:
        """
        Check if the current node is a root node.

        Returns:
            bool: True if the node is a root node, False otherwise.
        """
        return not self.parent_nodes

    @property
    def all_branches_keys(self) -> BranchKeyGeneratorP[BranchKey]:
        """
        Returns a generator that yields the branch keys for the current state.

        Returns:
            BranchKeyGenerator: A generator that yields the branch keys.
        """
        return self.state_.branch_keys

    def add_parent(self, branch_key: BranchKey, new_parent_node: FamilyT) -> None:
        """
        Adds a new parent node to the current node.

        Args:
            branch_key (BranchKey): The branch key that led to the node from the new parent node.
            new_parent_node (ITreeNode): The new parent node to be added.

        Raises:
            AssertionError: If the new parent node is already in the parent nodes set.

        Returns:
            None
        """
        # debug
        assert (
            new_parent_node not in self.parent_nodes
        )  # there cannot be two ways to link the same child-parent
        self.parent_nodes[new_parent_node] = branch_key

    def is_over(self) -> bool:
        """
        Checks if the state is terminal.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return self.state.is_game_over()

    def print_branches_children(self) -> None:
        """
        Prints the branches-children link of the node.

        This method prints the branches-children link of the node, showing the branch and the ID of the child node.
        If a child node is None, it will be displayed as 'None'.

        Returns:
            None
        """
        print(
            "here are the ",
            len(self.branches_children_),
            " branches-children link of node",
            self.id,
            ": ",
            end=" ",
        )
        for branch, child in self.branches_children_.items():
            if child is None:
                print(branch, child, end=" ")
            else:
                print(branch, child.id, end=" ")
        print(" ")

    def dot_description(self) -> str:
        """
        Returns a string representation of the node in the DOT format.

        The string includes the node's ID, depth, and state tag.

        Returns:
            A string representation of the node in the DOT format.
        """
        return (
            "id:"
            + str(self.id)
            + " dep: "
            + str(self.tree_depth)
            + "\nfen:"
            + str(self.state.tag)
        )
