"""
Tree
"""

from typing import Any

from anemone.basics import TreeDepth
from anemone.nodes.itree_node import ITreeNode

from .descendants import RangedDescendants


class Tree[NodeT: ITreeNode[Any]]:
    """
    Represents a game tree of reachable states from a starting position.

    The root node contains the starting state. Each node contains a state and has
    children for each available branch. The tree tracks descendants and expansion
    counters for bookkeeping.
    """

    _root_node: NodeT
    descendants: RangedDescendants[NodeT]
    tree_root_tree_depth: TreeDepth

    def __init__(self, root_node: NodeT, descendants: RangedDescendants[NodeT]) -> None:
        """
        Initialize the Tree with a root node and descendants.

        Args:
            root_node: The root node of the tree.
            descendants: The descendants collection for the tree.
        """
        self.tree_root_tree_depth = root_node.tree_depth

        # number of nodes in the tree (already one as we have the root node provided)
        self.nodes_count = 1

        # integer counting the number of branches in the tree.
        # the interest of self.branch_count over the number of nodes in the descendants
        # is that is always increasing at each opening,
        # while self.node_count can stay the same if the nodes already existed.
        self.branch_count = 0

        self._root_node = root_node
        self.descendants = descendants

    @property
    def root_node(self) -> NodeT:
        """
        Returns the root node of the tree.

        Returns:
            NodeT: The root node of the tree.
        """
        return self._root_node

    def node_depth(self, node: NodeT) -> int:
        """
        Calculates the depth of a given node in the tree.

        Args:
            node: The node for which to calculate the depth.

        Returns:
            int: The depth of the node.
        """
        return node.tree_depth - self.tree_root_tree_depth
