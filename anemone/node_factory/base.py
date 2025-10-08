"""
Basic class for Creating Tree nodes
"""

from anemone.basics import BranchKey, NodeContent
from anemone.node_factory import TreeNodeFactory
from anemone.nodes.itree_node import ITreeNode
from anemone.nodes.tree_node import TreeNode


class Base[T: ITreeNode = ITreeNode](TreeNodeFactory[T]):
    """
    Basic class for Creating Tree nodes
    """

    def create(
        self,
        content: NodeContent,
        tree_depth: int,
        count: int,
        parent_node: ITreeNode | None,
        branch_from_parent: BranchKey | None,
    ) -> TreeNode[T]:
        """
        Creates a new TreeNode object.

        Args:
            board (boards.BoardChi): The current board state.
            half_move (int): The half-move count.
            count (int): The ID of the new node.
            parent_node (ITreeNode | None): The parent node of the new node.
            move_from_parent (chess.Move | None): The move that leads to the new node.

        Returns:
            TreeNode: The newly created TreeNode object.
        """

        parent_nodes: dict[ITreeNode, BranchKey]
        if parent_node is None:
            parent_nodes = {}
        else:
            assert branch_from_parent is not None
            parent_nodes = {parent_node: branch_from_parent}

        tree_node: TreeNode[T] = TreeNode(
            content_=content,
            tree_depth_=tree_depth,
            id_=count,
            parent_nodes_=parent_nodes,
        )
        return tree_node
