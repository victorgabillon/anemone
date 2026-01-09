"""
Tree expansion representations for managing game trees.
"""

import typing
from dataclasses import dataclass, field

from valanga import BranchKey, StateModifications

from anemone import trees
import anemone.nodes as node

NodeT = typing.TypeVar("NodeT", bound=node.ITreeNode[typing.Any])


@dataclass(slots=True)
class TreeExpansion[NodeT: node.ITreeNode[typing.Any] = node.ITreeNode[typing.Any]]:
    """
    Represents an expansion of a tree in a chess game.

    Attributes:
        child_node (NodeT): The child node created during the expansion.
        parent_node (node.ITreeNode | None): The parent node of the child node. None if it's the root node.
        board_modifications (board_mod.BoardModification | None): The modifications made to the chess board during the expansion.
        creation_child_node (bool): Indicates whether the child node was created during the expansion.
        move (chess.Move): the move from parent to child node.
    """

    child_node: NodeT
    parent_node: NodeT | None
    state_modifications: StateModifications | None
    creation_child_node: bool
    branch_key: BranchKey | None

    def __repr__(self) -> str:
        return (
            f"child_node{self.child_node.id} | "
            f"parent_node{self.parent_node.id if self.parent_node is not None else None} | "
            f"creation_child_node{self.creation_child_node}"
        )


def record_tree_expansion(
    *,
    tree: trees.Tree[NodeT],
    tree_expansions: "TreeExpansions[NodeT]",
    tree_expansion: TreeExpansion[NodeT],
) -> None:
    """Apply one TreeExpansion's bookkeeping to a tree + its log."""

    if tree_expansion.creation_child_node:
        tree.nodes_count += 1
        tree.descendants.add_descendant(tree_expansion.child_node)

    tree_expansions.add(tree_expansion=tree_expansion)

def _new_expansions_list() -> list[TreeExpansion[typing.Any]]:
    return []


@dataclass(slots=True)
class TreeExpansions[NodeT: node.ITreeNode[typing.Any] = node.ITreeNode[typing.Any]]:
    """
    Represents a collection of tree expansions in a chess game.

    Attributes:
        expansions_with_node_creation (List[TreeExpansion]): List of expansions where child nodes were created.
        expansions_without_node_creation (List[TreeExpansion]): List of expansions where child nodes were not created.
    """

    expansions_with_node_creation: list[TreeExpansion[NodeT]] = field(default_factory=_new_expansions_list)
    expansions_without_node_creation: list[TreeExpansion[NodeT]] = field(default_factory=_new_expansions_list)



    def __iter__(self) -> typing.Iterator[TreeExpansion[NodeT]]:
        return iter(
            self.expansions_with_node_creation + self.expansions_without_node_creation
        )

    def add(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """
        Adds a tree expansion to the collection.

        Args:
            tree_expansion (TreeExpansion): The tree expansion to add.
        """
        if tree_expansion.creation_child_node:
            self.add_creation(tree_expansion=tree_expansion)
        else:
            self.add_connection(tree_expansion=tree_expansion)

    def add_creation(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """
        Adds a tree expansion with a created child node to the collection.

        Args:
            tree_expansion (TreeExpansion): The tree expansion to add.
        """
        self.expansions_with_node_creation.append(tree_expansion)

    def add_connection(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """
        Adds a tree expansion without a created child node to the collection.

        Args:
            tree_expansion (TreeExpansion): The tree expansion to add.
        """
        self.expansions_without_node_creation.append(tree_expansion)

    def __str__(self) -> str:
        return (
            f"expansions_with_node_creation {self.expansions_with_node_creation} \n"
            f"expansions_without_node_creation{self.expansions_without_node_creation}"
        )
