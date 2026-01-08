"""
This module contains the definition of the NodeSelector class and related types.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import anemone.trees as trees
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

from .opening_instructions import OpeningInstructions

if TYPE_CHECKING:
    import anemone.tree_manager as tree_man


@dataclass
class NodeSelectorState:
    """Node Selector State"""

    ...


class NodeSelector[TNode: AlgorithmNode[Any] = AlgorithmNode[Any]](Protocol):
    """
    Protocol for Node Selectors.
    """

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[TNode],
        latest_tree_expansions: "tree_man.TreeExpansions[TNode]",
    ) -> OpeningInstructions[TNode]:
        """
        Selects a node from the given tree and returns the instructions to move to an open position.

        Args:
            tree: The tree containing the nodes.
            latest_tree_expansions: The latest expansions of the tree.

        Returns:
            OpeningInstructions: The instructions to move to an open position.
        """
        ...
