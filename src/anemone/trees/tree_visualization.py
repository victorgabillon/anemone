"""
This module provides functions for visualizing and saving tree structures.

The functions in this module allow for the visualization of tree structures using the Graphviz library.
It provides a way to display the tree structure as a graph and save it as a PDF file.
Additionally, it provides a function to save the raw data of the tree structure to a file using pickle.

Functions:
- add_dot(dot: Digraph, treenode: ITreeNode) -> None: Adds nodes and edges to the graph representation of the tree.
- display_special(node: ITreeNode, format: str, index: dict[chess.Move, str]) -> Digraph: Displays a special
representation of the tree with additional information.
- display(tree: ValueTree, format_str: str) -> Digraph: Displays the tree structure as a graph.
- save_pdf_to_file(tree: ValueTree) -> None: Saves the tree structure as a PDF file.
- save_raw_data_to_file(tree: ValueTree, count: str = '#') -> None: Saves the raw data of the tree
structure to a file.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import sys
from pickle import dump

from graphviz import Digraph
from valanga import BranchKey, State

from anemone.nodes import ITreeNode
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from .tree import Tree


def add_dot[StateT: State](dot: Digraph, treenode: ITreeNode[StateT]) -> None:
    """
    Adds a node and edges to the given Dot graph based on the provided tree node.

    Args:
        dot (Digraph): The Dot graph to add the node and edges to.
        treenode (ITreeNode): The tree node to visualize.

    Returns:
        None
    """
    nd = treenode.dot_description()
    dot.node(str(treenode.id), nd)
    branch: BranchKey
    for _, branch in enumerate(treenode.branches_children):
        if treenode.branches_children[branch] is not None:
            child = treenode.branches_children[branch]
            if child is not None:
                cdd = str(child.id)
                dot.edge(
                    str(treenode.id),
                    cdd,
                    str(treenode.state.branch_name_from_key(key=branch)),
                )
                add_dot(dot, child)


def display_special[StateT: State](
    node: AlgorithmNode[StateT],  # or AlgorithmNode if you prefer
    format_str: str,
    index: dict[BranchKey, str],
) -> Digraph:
    """Display a tree with custom edge labels for the given node."""
    dot = Digraph(format=format_str)

    nd = node.dot_description()
    dot.node(str(node.id), nd)

    sorted_branches: list[BranchKey] = sorted(node.branches_children.keys(), key=str)

    for branch_key in sorted_branches:
        child = node.branches_children[branch_key]
        if child is None:
            continue

        edge_description: str = (
            index[branch_key]
            + "|"
            + str(node.state.branch_name_from_key(key=branch_key))
            + "|"
            + node.tree_evaluation.description_tree_visualizer_branch(child)
        )
        dot.edge(str(node.id), str(child.id), edge_description)
        dot.node(str(child.id), child.dot_description())
        print("--branch:", edge_description)
        print("--child:", child.dot_description())

    return dot


def display[StateT: State](
    tree: Tree[AlgorithmNode[StateT]], format_str: str
) -> Digraph:
    """
    Display the move and value tree using graph visualization.

    Args:
        tree (Tree): The move and value tree to be displayed.
        format_str (str): The format of the output graph (e.g., 'png', 'pdf', 'svg').

    Returns:
        Digraph: The graph representation of the move and value tree.
    """
    dot = Digraph(format=format_str)
    add_dot(dot, tree.root_node)
    return dot


def save_pdf_to_file[StateT: State](tree: Tree[AlgorithmNode[StateT]]) -> None:
    """
    Saves the visualization of a tree as a PDF file.

    Args:
        tree (Tree): The tree to be visualized and saved.

    Returns:
        None
    """
    dot = display(tree=tree, format_str="pdf")
    tag_ = tree.root_node.state.tag
    dot.render("chipiron/runs/treedisplays/TreeVisual_" + str(tag_) + ".pdf")


def save_raw_data_to_file(tree: Tree[AlgorithmNode], count: str = "#") -> None:
    """
    Save raw data of a ValueTree to a file.

    Args:
        tree (Tree): The Tree object to save.
        count (str, optional): A string to append to the filename. Defaults to '#'.

    Returns:
        None
    """
    tag_ = tree.root_node.state.tag
    filename = "chipiron/debugTreeData_" + str(tag_) + "-" + str(count) + ".td"

    sys.setrecursionlimit(100000)
    with open(filename, "wb") as f:
        dump([tree.descendants, tree.root_node], f)
