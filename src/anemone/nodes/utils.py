"""
This module contains utility functions for working with tree nodes in the selector.

Functions:
- are_all_moves_and_children_opened(tree_node: TreeNode) -> bool: Checks if all branches and children are opened.
- a_branch_key_sequence_from_root(tree_node: ITreeNode) -> list[str]: Returns a list of branch keys from the root to a given node.
- print_a_move_sequence_from_root(tree_node: TreeNode) -> None: Prints the branch sequence from the root to a given node.
- is_winning(node_tree_evaluation: NodeTreeEvaluation, color: Color) -> bool: Checks if the side to act in the node is winning.
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
    Checks if all branches and children of a tree node are opened.

    Args:
        tree_node (TreeNode): The tree node to check.

    Returns:
        bool: True if all branches and children are opened, False otherwise.
    """
    return tree_node.all_branches_generated and tree_node.non_opened_branches == set()


def a_branch_key_sequence_from_root[StateT: State](
    tree_node: ITreeNode[StateT],
) -> list[str]:
    """
    Returns a list of branch sequences from the root node to a given tree node.

    Args:
        tree_node (ITreeNode): The tree node to get the branch sequence for.

    Returns:
        list[str]: A list of branch sequences from the root node to the given tree node.
    """
    branch_sequence_from_root: list[BranchKey] = []
    child: ITreeNode[StateT] = tree_node
    while child.parent_nodes:
        parent: ITreeNode[StateT] = next(iter(child.parent_nodes))
        branch: BranchKey = child.parent_nodes[parent]
        branch_sequence_from_root.append(branch)
        child = parent
    branch_sequence_from_root.reverse()
    return [str(i) for i in branch_sequence_from_root]


def a_branch_str_sequence_from_root[StateT: State](
    tree_node: ITreeNode[StateT],
) -> list[str]:
    """
    Returns a list of branch sequences from the root node to a given tree node.

    Args:
        tree_node (ITreeNode): The tree node to get the branch sequence for.

    Returns:
        list[str]: A list of branch sequences from the root node to the given tree node.
    """
    branch_sequence_from_root: list[str] = []
    child: ITreeNode[StateT] = tree_node
    while child.parent_nodes:
        parent: ITreeNode[StateT] = next(iter(child.parent_nodes))
        branch_key: BranchKey = child.parent_nodes[parent]
        branch_str: str = parent.state.branch_name_from_key(branch_key)
        branch_sequence_from_root.append(branch_str)
        child = parent
    branch_sequence_from_root.reverse()
    return [str(i) for i in branch_sequence_from_root]


def best_node_sequence_from_node[StateT: State](
    tree_node: AlgorithmNode[StateT],
) -> list[AlgorithmNode[StateT]]:
    """
    Returns the best node sequence from the given tree node following the best branches.
    Args:
        tree_node (AlgorithmNode): The tree node to start from.
    Returns:
        list[AlgorithmNode]: A list of tree nodes representing the best branch sequence.
    """

    best_move_seq: list[BranchKey] = tree_node.tree_evaluation.best_branch_sequence
    index = 0
    move_sequence: list[AlgorithmNode[StateT]] = [tree_node]
    child: AlgorithmNode[StateT] = tree_node
    while child.branches_children:
        move: BranchKey = best_move_seq[index]
        child_ = child.branches_children[move]
        assert child_ is not None
        child = child_
        move_sequence.append(child)
        index = index + 1
    return move_sequence


def print_a_move_sequence_from_root[StateT: State](
    tree_node: ITreeNode[StateT],
) -> None:
    """
    Prints the branch sequence from the root node to a given tree node.

    Args:
        tree_node (TreeNode): The tree node to print the branch sequence for.

    Returns:
        None
    """
    move_sequence_from_root: list[str] = a_branch_key_sequence_from_root(
        tree_node=tree_node
    )
    print(f"a_move_sequence_from_root{move_sequence_from_root}")


def is_winning(node_tree_evaluation: NodeTreeEvaluation, color: Color) -> bool:
    """
    Checks if the color to play in the node is winning.

    Args:
        node_tree_evaluation: The evaluation of the node.
        color: The color to check.

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
