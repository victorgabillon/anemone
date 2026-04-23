"""Basic class for Creating Tree nodes."""

from typing import Any, Protocol

from valanga import BranchKey, State, StateModifications

from anemone.basics import TreeDepth
from anemone.nodes.itree_node import ITreeNode
from anemone.nodes.state_handles import MaterializedStateHandle, StateHandle
from anemone.nodes.tree_node import TreeNode


class NodeFactory[
    NodeT: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
](Protocol):
    """Node Factory."""

    def create(
        self,
        state_handle: StateHandle[StateT],
        tree_depth: TreeDepth,
        count: int,
        parent_node: NodeT | None,
        branch_from_parent: BranchKey | None,
        modifications: StateModifications | None,
    ) -> NodeT:
        """Create a node from a state handle and tree metadata."""
        ...


class TreeNodeFactory[T: ITreeNode[Any] = ITreeNode[Any], StateT: State = State]:
    """Basic class for Creating Tree nodes."""

    def create(
        self,
        state_handle: StateHandle[StateT],
        tree_depth: TreeDepth,
        count: int,
        parent_node: T | None,
        branch_from_parent: BranchKey | None,
        modifications: StateModifications | None = None,
    ) -> TreeNode[T, StateT]:
        """Create a new TreeNode object.

        Args:
            state_handle: The explicit state handle for the node.
            tree_depth: The tree depth for the new node.
            count: The ID of the new node.
            parent_node: The parent node of the new node.
            branch_from_parent: The branch key from the parent node.
            modifications: The state modifications, if any.

        Returns:
            The newly created TreeNode object.

        """
        # TreeNode doesn't use modifications (it's a pure data container).
        _ = modifications

        parent_nodes: dict[T, set[BranchKey]]
        if parent_node is None:
            parent_nodes = {}
        else:
            assert branch_from_parent is not None
            parent_nodes = {parent_node: {branch_from_parent}}

        tree_node: TreeNode[T, StateT] = TreeNode[T, StateT](
            state_handle_=state_handle,
            tree_depth_=tree_depth,
            id_=count,
            parent_nodes_=parent_nodes,
        )
        return tree_node

    def create_from_state(
        self,
        state: StateT,
        tree_depth: TreeDepth,
        count: int,
        parent_node: T | None,
        branch_from_parent: BranchKey | None,
        modifications: StateModifications | None = None,
    ) -> TreeNode[T, StateT]:
        """Create a tree node from a concrete state."""
        return self.create(
            state_handle=MaterializedStateHandle(state_=state),
            tree_depth=tree_depth,
            count=count,
            parent_node=parent_node,
            branch_from_parent=branch_from_parent,
            modifications=modifications,
        )
