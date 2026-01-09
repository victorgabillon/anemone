"""
This module contains utility functions for working with tree nodes in the move selector.

Functions:
- are_all_moves_and_children_opened(tree_node: TreeNode) -> bool: Checks if all moves and children of a tree node are opened.
- a_move_sequence_from_root(tree_node: ITreeNode) -> list[str]: Returns a list of move sequences from the root node to a given tree node.
- print_a_move_sequence_from_root(tree_node: TreeNode) -> None: Prints the move sequence from the root node to a given tree node.
- is_winning(node_minmax_evaluation: NodeMinmaxEvaluation, color: chess.Color) -> bool: Checks if the color to play in the node is winning.
"""

from valanga import BranchKey, Color, State

from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation import (
    NodeTreeEvaluation,
)
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from .itree_node import ITreeNode
from .tree_node import TreeNode


def are_all_moves_and_children_opened(tree_node: TreeNode) -> bool:
    """
    Checks if all moves and children of a tree node are opened.

    Args:
        tree_node (TreeNode): The tree node to check.

    Returns:
        bool: True if all moves and children are opened, False otherwise.
    """
    return tree_node.all_branches_generated and tree_node.non_opened_branches == set()


def a_move_key_sequence_from_root[TState: State](
    tree_node: ITreeNode[TState],
) -> list[str]:
    """
    Returns a list of move sequences from the root node to a given tree node.

    Args:
        tree_node (ITreeNode): The tree node to get the move sequence for.

    Returns:
        list[str]: A list of move sequences from the root node to the given tree node.
    """
    move_sequence_from_root: list[BranchKey] = []
    child: ITreeNode[TState] = tree_node
    while child.parent_nodes:
        parent: ITreeNode[TState] = next(iter(child.parent_nodes))
        move: BranchKey = child.parent_nodes[parent]
        move_sequence_from_root.append(move)
        child = parent
    move_sequence_from_root.reverse()
    return [str(i) for i in move_sequence_from_root]


def a_branch_str_sequence_from_root[TState: State](
    tree_node: ITreeNode[TState],
) -> list[str]:
    """
    Returns a list of move sequences from the root node to a given tree node.

    Args:
        tree_node (ITreeNode): The tree node to get the move sequence for.

    Returns:
        list[str]: A list of move sequences from the root node to the given tree node.
    """
    move_sequence_from_root: list[str] = []
    child: ITreeNode[TState] = tree_node
    while child.parent_nodes:
        parent: ITreeNode[TState] = next(iter(child.parent_nodes))
        branch_key: BranchKey = child.parent_nodes[parent]
        branch_str: str = parent.state.branch_name_from_key(branch_key)
        move_sequence_from_root.append(branch_str)
        child = parent
    move_sequence_from_root.reverse()
    return [str(i) for i in move_sequence_from_root]


def best_node_sequence_from_node[TState: State](
    tree_node: AlgorithmNode[TState],
) -> list[AlgorithmNode[TState]]:
    """ """

    best_move_seq: list[BranchKey] = tree_node.tree_evaluation.best_branch_sequence
    index = 0
    move_sequence: list[AlgorithmNode[TState]] = [tree_node]
    child: AlgorithmNode[TState] = tree_node
    while child.branches_children:
        move: BranchKey = best_move_seq[index]
        child_ = child.branches_children[move]
        assert child_ is not None
        child = child_
        move_sequence.append(child)
        index = index + 1
    return move_sequence


def print_a_move_sequence_from_root[TState: State](
    tree_node: ITreeNode[TState],
) -> None:
    """
    Prints the move sequence from the root node to a given tree node.

    Args:
        tree_node (TreeNode): The tree node to print the move sequence for.

    Returns:
        None
    """
    move_sequence_from_root: list[str] = a_move_key_sequence_from_root(
        tree_node=tree_node
    )
    print(f"a_move_sequence_from_root{move_sequence_from_root}")


def is_winning(node_tree_evaluation: NodeTreeEvaluation, color: Color) -> bool:
    """
    Checks if the color to play in the node is winning.

    Args:
        node_minmax_evaluation (NodeMinmaxEvaluation): The evaluation of the node.
        color (chess.Color): The color to check.

    Returns:
        bool: True if the color is winning, False otherwise.
    """
    assert node_tree_evaluation.value_white_minmax is not None
    winning_if_color_white: bool = (
        node_tree_evaluation.value_white_minmax > 0.98 and color is Color.WHITE
    )
    winning_if_color_black: bool = (
        node_tree_evaluation.value_white_minmax < -0.98 and color is Color.BLACK
    )

    return winning_if_color_white or winning_if_color_black
