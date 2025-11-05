"""
This module contains the definition of the NodeSelector class and related types.
"""

import typing
from dataclasses import dataclass
from typing import Protocol

from ..trees import ValueTree
from .opening_instructions import OpeningInstructions

if typing.TYPE_CHECKING:
    import anemone.tree_manager as tree_man


@dataclass
class NodeSelectorState:
    """Node Selector State"""

    ...


class NodeSelector(Protocol):
    """
    Protocol for Node Selectors.
    """

    def choose_node_and_move_to_open(
        self, tree: ValueTree, latest_tree_expansions: "tree_man.TreeExpansions"
    ) -> OpeningInstructions:
        """
        Selects a node from the given tree and returns the instructions to move to an open position.

        Args:
            tree (ValueTree): The tree containing the moves and their corresponding values.
            latest_tree_expansions (tree_man.TreeExpansions): The latest expansions of the tree.

        Returns:
            OpeningInstructions: The instructions to move to an open position.
        """
        ...
