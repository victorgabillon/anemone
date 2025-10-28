"""
TreeNodeFactory Protocol
"""

from typing import Protocol

from valanga import BranchKey, State

import anemone.nodes as node
from anemone.basics import TreeDepth
from anemone.nodes.itree_node import ITreeNode


class TreeNodeFactory[T: ITreeNode](Protocol):
    """
    Interface for Tree Node Factories
    """

    def create(
        self,
        state: State,
        tree_depth: TreeDepth,
        count: int,
        parent_node: node.ITreeNode | None,
        branch_from_parent: BranchKey | None,
    ) -> node.ITreeNode[T]:
        """
        Creates a new TreeNode object.

        Args:
            board (boards.BoardChi): The current state of the chess board.
            half_move (int): The number of half moves made so far in the game.
            count (int): The number of times this position has occurred in the game.
            parent_node (node.ITreeNode | None): The parent node of the new TreeNode.
            move_from_parent (chess.Move | None): The move that led to this position, if any.
            modifications (board_mod.BoardModification | None): The modifications made to the board, if any.

        Returns:
            node.TreeNode: The newly created TreeNode object.
        """
        ...
