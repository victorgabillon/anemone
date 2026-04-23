"""Generic structural node implementation shared by tree wrappers."""

from dataclasses import dataclass, field
from typing import Any

from valanga import BranchKey, State, StateTag

from .itree_node import ITreeNode
from .state_handles import StateHandle

# TODO: replace the any with a defaut value in ITReenode when availble in python; 3.13?


@dataclass(slots=True)
class TreeNode[
    FamilyT: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
]:
    r"""Concrete structural implementation of ``ITreeNode``.

    ``TreeNode`` stores navigation and branch-opening bookkeeping only:
    state-handle ownership, tree depth, parent/child links, and unopened-branch
    tracking. Search/runtime layers should wrap it rather than grow more
    algorithm semantics here.

    Attributes:
        id\_ (int): The number to identify this node for easier debugging.
        tree_depth\_ (int): The depth of the node in the tree.
        state\_handle\_ (StateHandle[StateT]): The handle for the node state.
        parent_nodes\_ (dict[ITreeNode, set[BranchKey]]): Parent nodes and the
            distinct branch keys linking them to this node.
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
        add_parent(new_parent_node: ITreeNode): Adds a parent node to the current node.
        is_over(): Checks if the state is terminal.
        print_branches_children(): Prints the branches-children links of the node.

    """

    # id is a number to identify this node for easier debug
    id_: int

    # the tree depth of this node
    tree_depth_: int

    # the node holds a state handle.
    state_handle_: StateHandle[StateT]

    # Each parent can reach this node through multiple distinct branch keys.
    parent_nodes_: dict[FamilyT, set[BranchKey]]

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

    # dictionary mapping branches to children nodes. Node is set to None if not created
    branches_children_: dict[BranchKey, FamilyT | None] = field(
        default_factory=_empty_branches_children
    )

    @property
    def tag(self) -> StateTag:
        """Returns the fast tag representation of the state.

        Returns:
            StateTag: The fast tag representation of the state.

        """
        return self.state.tag

    @property
    def id(self) -> int:
        """Returns the ID of the tree node.

        Returns:
            int: The ID of the tree node.

        """
        return self.id_

    @property
    def state(self) -> StateT:
        """Returns the state associated with this tree node.

        Returns:
            State: The state associated with this tree node.

        """
        return self.state_handle_.get()

    @property
    def state_handle(self) -> StateHandle[StateT]:
        """Return the explicit state handle owned by this node."""
        return self.state_handle_

    @property
    def tree_depth(self) -> int:
        """Returns the tree depth of this node.

        Returns:
            int: The tree depth of this node.

        """
        return self.tree_depth_

    @property
    def branches_children(self) -> dict[BranchKey, FamilyT | None]:
        """Return a bidirectional dictionary of children nodes for the current tree node.

        This includes the corresponding branches that lead to each child node.

        Returns:
            dict[BranchKey, ITreeNode | None]: A bidirectional dictionary mapping branches to
            the corresponding child nodes. If a branch does not have a corresponding child node, it is
            mapped to None.

        """
        return self.branches_children_

    @property
    def parent_nodes(self) -> dict[FamilyT, set[BranchKey]]:
        """Return the incoming parent-edge mapping for this node.

        Each key is a parent node. Each value is the set of distinct branch keys
        through which that parent reaches this node.
        """
        return self.parent_nodes_

    def is_root_node(self) -> bool:
        """Check if the current node is a root node.

        Returns:
            bool: True if the node is a root node, False otherwise.

        """
        return not self.parent_nodes

    def add_parent(self, branch_key: BranchKey, new_parent_node: FamilyT) -> None:
        """Add a new parent node to the current node.

        Args:
            branch_key (BranchKey): The branch key that led to the node from the new parent node.
            new_parent_node (ITreeNode): The new parent node to be added.

        Raises:
            AssertionError: If the parent/branch edge already exists.

        Returns:
            None

        """
        branch_keys = self.parent_nodes.setdefault(new_parent_node, set())
        assert branch_key not in branch_keys, (
            f"Duplicate parent edge for child {self.id} from parent "
            f"{new_parent_node.id} via branch {branch_key!r}"
        )
        branch_keys.add(branch_key)

    def is_over(self) -> bool:
        """Check if the state is terminal.

        Returns:
            bool: True if the state is terminal, False otherwise.

        """
        return self.state.is_game_over()

    def print_branches_children(self) -> None:
        """Print the branches-children link of the node.

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
