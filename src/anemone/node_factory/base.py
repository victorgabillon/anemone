"""
Basic class for Creating Tree nodes
"""

from typing import Any, Protocol

from valanga import BranchKey, State, StateModifications

from anemone.basics import TreeDepth
from anemone.nodes.itree_node import ITreeNode
from anemone.nodes.tree_node import TreeNode


class NodeFactory[NodeT: ITreeNode[Any] = ITreeNode[Any]](Protocol):
    """
    Node Factory
    """

    def create(
        self,
        state: State,
        tree_depth: TreeDepth,
        count: int,
        parent_node: NodeT | None,
        branch_from_parent: BranchKey | None,
        modifications: StateModifications | None,
    ) -> NodeT:
        """Create a node from state and tree metadata."""
        ...


class TreeNodeFactory[T: ITreeNode[Any] = ITreeNode[Any], StateT: State = State]:
    """
    Basic class for Creating Tree nodes
    """

    def create(
        self,
        state: StateT,
        tree_depth: TreeDepth,
        count: int,
        parent_node: T | None,
        branch_from_parent: BranchKey | None,
        modifications: StateModifications | None = None,
    ) -> TreeNode[T, StateT]:
        """
        Create a new TreeNode object.

        Args:
            state: The current state for the node.
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

        parent_nodes: dict[T, BranchKey]
        if parent_node is None:
            parent_nodes = {}
        else:
            assert branch_from_parent is not None
            parent_nodes = {parent_node: branch_from_parent}

        tree_node: TreeNode[T, StateT] = TreeNode[T, StateT](
            state_=state,
            tree_depth_=tree_depth,
            id_=count,
            parent_nodes_=parent_nodes,
        )
        return tree_node
