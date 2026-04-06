"""Utility functions for tree nodes."""

from typing import Any

from valanga import BranchKey, Color, State

from anemone.dynamics import SearchDynamics
from anemone.node_evaluation.common.node_value_evaluation import NodeValueEvaluation
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from .itree_node import ITreeNode


def _representative_parent[StateT: State](
    child: ITreeNode[StateT],
) -> ITreeNode[StateT]:
    """Return one deterministic representative parent for ``child``."""
    assert child.parent_nodes
    return min(child.parent_nodes, key=lambda parent: (parent.id, repr(parent)))


def _representative_branch_key(branch_keys: set[BranchKey]) -> BranchKey:
    """Return one deterministic representative branch key."""
    assert branch_keys
    return min(branch_keys, key=repr)


def _representative_parent_and_branch_key[StateT: State](
    child: ITreeNode[StateT],
) -> tuple[ITreeNode[StateT], BranchKey]:
    """Select one representative incoming edge for ``child``."""
    parent = _representative_parent(child)
    branch_key = _representative_branch_key(child.parent_nodes[parent])
    return parent, branch_key


def are_all_branches_and_children_opened(tree_node: ITreeNode[Any]) -> bool:
    """Check if all structural branches of a node have been opened.

    Args:
        tree_node (ITreeNode): The tree node to check.

    Returns:
        bool: True if all branches and children are opened, False otherwise.

    """
    return tree_node.all_branches_generated and tree_node.non_opened_branches == set()


def a_branch_key_sequence_from_root[StateT: State](
    tree_node: ITreeNode[StateT],
) -> list[str]:
    """Return a representative branch-key sequence from the root to this node.

    If the structure is a DAG or multiedge graph, there may be multiple valid
    root-to-node paths. This helper selects one representative incoming edge at
    each step.

    Args:
        tree_node (ITreeNode): The tree node to get a representative path for.

    Returns:
        list[str]: A representative branch-key sequence from the root node to
        the given tree node.

    """
    branch_sequence_from_root: list[BranchKey] = []
    child: ITreeNode[StateT] = tree_node
    while child.parent_nodes:
        parent, branch = _representative_parent_and_branch_key(child)
        branch_sequence_from_root.append(branch)
        child = parent
    branch_sequence_from_root.reverse()
    return [str(i) for i in branch_sequence_from_root]


def a_branch_str_sequence_from_root[StateT: State](
    tree_node: ITreeNode[StateT], dynamics: SearchDynamics[StateT, Any]
) -> list[str]:
    """Return a representative branch-label sequence from the root to this node.

    If the structure is a DAG or multiedge graph, there may be multiple valid
    root-to-node paths. This helper selects one representative incoming edge at
    each step.

    Args:
        tree_node (ITreeNode): The tree node to get a representative path for.
        dynamics (SearchDynamics): The dynamics used for labeling the edges in the visualization.

    Returns:
        list[str]: A representative branch-label sequence from the root node to
        the given tree node.

    """
    branch_sequence_from_root: list[str] = []
    child: ITreeNode[StateT] = tree_node
    while child.parent_nodes:
        parent, branch_key = _representative_parent_and_branch_key(child)
        branch_str: str = dynamics.action_name(parent.state, branch_key)
        branch_sequence_from_root.append(branch_str)
        child = parent
    branch_sequence_from_root.reverse()
    return [str(i) for i in branch_sequence_from_root]


def best_node_sequence_from_node[StateT: State](
    tree_node: AlgorithmNode[StateT],
) -> list[AlgorithmNode[StateT]]:
    """Return the best node sequence from the given tree node following the best branches.

    Args:
        tree_node (AlgorithmNode): The tree node to start from.

    Returns:
        list[AlgorithmNode]: A list of tree nodes representing the best branch sequence.

    """
    best_branch_seq: list[BranchKey] = tree_node.tree_evaluation.best_branch_sequence
    index = 0
    branch_sequence: list[AlgorithmNode[StateT]] = [tree_node]
    child: AlgorithmNode[StateT] = tree_node
    while child.branches_children:
        branch: BranchKey = best_branch_seq[index]
        child_ = child.branches_children[branch]
        assert child_ is not None
        child = child_
        branch_sequence.append(child)
        index = index + 1
    return branch_sequence


def print_a_branch_sequence_from_root[StateT: State](
    tree_node: ITreeNode[StateT],
) -> None:
    """Print a representative branch sequence from the root to a given node.

    Args:
        tree_node (ITreeNode): The tree node to print a representative path for.

    Returns:
        None

    """
    branch_sequence_from_root: list[str] = a_branch_key_sequence_from_root(
        tree_node=tree_node
    )
    print(f"a_representative_branch_sequence_from_root{branch_sequence_from_root}")


def is_winning(node_tree_evaluation: NodeValueEvaluation, color: Color) -> bool:
    """Check if the color to play in the node is winning.

    Args:
        node_tree_evaluation: The evaluation of the node.
        color: The color to check.

    Returns:
        bool: True if the color is winning, False otherwise.

    """
    value = node_tree_evaluation.get_score()
    winning_if_color_white: bool = value > 0.98 and color is Color.WHITE
    winning_if_color_black: bool = value < -0.98 and color is Color.BLACK

    return winning_if_color_white or winning_if_color_black
