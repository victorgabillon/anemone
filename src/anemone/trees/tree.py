"""
Tree
"""

from anemone.basics import TreeDepth
from typing import Any

from anemone.nodes.itree_node import ITreeNode

from .descendants import RangedDescendants


class Tree[TNode: ITreeNode[Any]]:
    """
    This class defines the Tree that is built out of all the combinations of moves given a starting board position.
    The root node contains the starting board.
    Each node contains a board and has as many children node as there are legal move in the board.
    A children node then contains the board that is obtained by playing a particular moves in the board of the parent
    node.

    It is a pointer to the root node with some counters and keeping track of descendants.
    """

    _root_node: TNode
    descendants: RangedDescendants[TNode]
    tree_root_tree_depth: TreeDepth

    def __init__(
        self, root_node: TNode, descendants: RangedDescendants[TNode]
    ) -> None:
        """
        Initialize the Tree with a root node and descendants.

        Args:
            root_node: The root node of the tree.
            descendants: The descendants collection for the tree.
        """
        self.tree_root_tree_depth = root_node.tree_depth

        # number of nodes in the tree (already one as we have the root node provided)
        self.nodes_count = 1

        # integer counting the number of moves in the tree.
        # the interest of self.move_count over the number of nodes in the descendants
        # is that is always increasing at each opening,
        # while self.node_count can stay the same if the nodes already existed.
        self.move_count = 0

        self._root_node = root_node
        self.descendants = descendants

    @property
    def root_node(self) -> TNode:
        """
        Returns the root node of the tree.

        Returns:
            TNode: The root node of the tree.
        """
        return self._root_node

    def node_depth(self, node: TNode) -> int:
        """
        Calculates the depth of a given node in the tree.

        Args:
            node: The node for which to calculate the depth.

        Returns:
            int: The depth of the node.
        """
        return node.tree_depth - self.tree_root_tree_depth
