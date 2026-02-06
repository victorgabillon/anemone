"""This module contains the definition of the NodeSelector class and related types."""


from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from anemone import trees
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

from .opening_instructions import OpeningInstructions

if TYPE_CHECKING:
    from anemone import tree_manager as tree_man


@dataclass
class NodeSelectorState:
    """Node Selector State."""


class NodeSelector[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](Protocol):
    """Protocol for Node Selectors."""

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> OpeningInstructions[NodeT]:
        """Selects a node from the given tree and returns the instructions to open a branch.

        Args:
            tree: The tree containing the nodes.
            latest_tree_expansions: The latest expansions of the tree.

        Returns:
            OpeningInstructions: The instructions to open a branch.

        """
        raise NotImplementedError
