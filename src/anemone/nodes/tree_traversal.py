"""
This module provides functions for traversing a tree of nodes.

The functions in this module allow you to retrieve descendants of a given node in a tree structure.
"""

from typing import Any

from .algorithm_node import AlgorithmNode
from .itree_node import ITreeNode


def get_descendants[NodeT: ITreeNode[Any]](from_tree_node: NodeT) -> dict[NodeT, None]:
    """
    Get all descendants of a given tree node.

    Args:
        from_tree_node (ITreeNode): The starting tree node.

    Returns:
        dict[ITreeNode, None]: A dictionary containing all descendants of the starting tree node.
    """
    des: dict[NodeT, None] = {from_tree_node: None}  # include itself
    generation: set[NodeT] = {
        node for node in from_tree_node.branches_children.values() if node is not None
    }

    while generation:
        next_depth_generation: set[NodeT] = set()
        for node in generation:
            assert node is not None
            des[node] = None
            for next_generation_child in node.branches_children.values():
                if next_generation_child is not None:
                    next_depth_generation.add(next_generation_child)
        generation = next_depth_generation
    return des


def get_descendants_candidate_to_open[NodeT: AlgorithmNode[Any]](
    from_tree_node: NodeT, max_depth: int | None = None
) -> list[NodeT]:
    """
    Get descendants of a given tree node that are not over.

    Args:
        from_tree_node (AlgorithmNode): The starting tree node.
        max_depth (int | None, optional): The maximum depth to traverse. Defaults to None.

    Returns:
        list[AlgorithmNode]: A list of descendants that are not over.
    """
    if not from_tree_node.all_branches_generated and not from_tree_node.is_over():
        # should use are_all_branches_and_children_opened() but its messy!
        # also using is_over is  messy as over_events are defined in a child class!!!
        des = {from_tree_node: None}  # include itself maybe
    else:
        des = {}
    generation: set[NodeT] = {
        node for node in from_tree_node.branches_children.values() if node is not None
    }
    depth: int = 1
    assert max_depth is not None
    while generation and depth <= max_depth:
        next_depth_generation: set[NodeT] = set()
        for node in generation:
            if not node.all_branches_generated and not node.is_over():
                des[node] = None
            for next_generation_child in node.branches_children.values():
                if next_generation_child is not None:
                    next_depth_generation.add(next_generation_child)
        generation = next_depth_generation
    return list(des.keys())


def get_descendants_candidate_not_over[NodeT: AlgorithmNode[Any]](
    from_tree_node: NodeT, max_depth: int | None = None
) -> list[NodeT]:
    """
    Get descendants of a given tree node that are not over.

    Args:
        from_tree_node (ITreeNode): The starting tree node.
        max_depth (int | None, optional): The maximum depth to traverse. Defaults to None.

    Returns:
        list[ITreeNode]: A list of descendants that are not over.
    """
    assert not from_tree_node.is_over()
    if not from_tree_node.branches_children:
        return [from_tree_node]
    des: dict[NodeT, None] = {}
    generation: set[NodeT] = {
        node for node in from_tree_node.branches_children.values() if node is not None
    }

    depth: int = 1
    assert max_depth is not None
    while generation and depth <= max_depth:
        next_depth_generation: set[NodeT] = set()
        for node in generation:
            if not node.is_over():
                des[node] = None
            for next_generation_child in node.branches_children.values():
                if next_generation_child is not None:
                    next_depth_generation.add(next_generation_child)
        generation = next_depth_generation
    return list(des.keys())
