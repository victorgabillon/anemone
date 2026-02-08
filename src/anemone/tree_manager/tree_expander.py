"""Tree expansion representations for managing game trees."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

from valanga import BranchKey, StateModifications

from anemone import nodes as node
from anemone import trees

NodeT = TypeVar("NodeT", bound=node.ITreeNode[Any])


@dataclass(slots=True)
class TreeExpansion[NodeT: node.ITreeNode[Any] = node.ITreeNode[Any]]:
    """Represents a single tree expansion.

    Attributes:
        child_node: The child node connected or created during the expansion.
        parent_node: The parent node of the child node, or None for root entries.
        state_modifications: The state modifications produced by the transition.
        creation_child_node: Whether a new node was created during the expansion.
        branch_key: The branch key from parent to child.

    """

    child_node: NodeT
    parent_node: NodeT | None
    state_modifications: StateModifications | None
    creation_child_node: bool
    branch_key: BranchKey | None

    def __repr__(self) -> str:
        """Return a debug representation of the tree expansion."""
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


def _new_expansions_list() -> list[TreeExpansion[Any]]:
    """Return a new list for tree expansions."""
    return []


@dataclass(slots=True)
class TreeExpansions[NodeT: node.ITreeNode[Any] = node.ITreeNode[Any]]:
    """Represents a collection of tree expansions for bookkeeping.

    Attributes:
        expansions_with_node_creation: Expansions where child nodes were created.
        expansions_without_node_creation: Expansions where child nodes already existed.

    """

    expansions_with_node_creation: list[TreeExpansion[NodeT]] = field(
        default_factory=_new_expansions_list
    )
    expansions_without_node_creation: list[TreeExpansion[NodeT]] = field(
        default_factory=_new_expansions_list
    )

    def __iter__(self) -> Iterator[TreeExpansion[NodeT]]:
        """Iterate over all recorded expansions."""
        return iter(
            self.expansions_with_node_creation + self.expansions_without_node_creation
        )

    def add(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """Add a tree expansion to the collection.

        Args:
            tree_expansion: The tree expansion to add.

        """
        if tree_expansion.creation_child_node:
            self.add_creation(tree_expansion=tree_expansion)
        else:
            self.add_connection(tree_expansion=tree_expansion)

    def add_creation(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """Add a tree expansion with a newly created child node.

        Args:
            tree_expansion: The tree expansion to add.

        """
        self.expansions_with_node_creation.append(tree_expansion)

    def add_connection(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """Add a tree expansion that only connects to an existing node.

        Args:
            tree_expansion: The tree expansion to add.

        """
        self.expansions_without_node_creation.append(tree_expansion)

    def __str__(self) -> str:
        """Return a string summary of recorded expansions."""
        return (
            f"expansions_with_node_creation {self.expansions_with_node_creation} \n"
            f"expansions_without_node_creation{self.expansions_without_node_creation}"
        )
