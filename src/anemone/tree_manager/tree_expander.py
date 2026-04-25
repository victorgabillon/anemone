"""Tree expansion representations for managing game trees."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from valanga import BranchKey, StateModifications

from anemone import nodes as node
from anemone import trees
from anemone.trees.debug_invariants import (
    DESCENDANTS_INVARIANT_LOG_PREFIX,
    descendants_invariant_debug_enabled,
    validate_descendants_tags,
)
from anemone.utils.logger import anemone_logger


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


def record_tree_expansion[NodeT: node.ITreeNode[Any]](
    *,
    tree: trees.Tree[NodeT],
    tree_expansions: "TreeExpansions[NodeT]",
    tree_expansion: TreeExpansion[NodeT],
) -> None:
    """Apply one TreeExpansion's bookkeeping to a tree + its log."""
    if tree_expansion.creation_child_node:
        if descendants_invariant_debug_enabled():
            _log_expansion_node(
                tree=tree,
                tree_expansion=tree_expansion,
                phase="before_register_child",
            )
        tree.nodes_count += 1
        tree.descendants.register_descendant(tree_expansion.child_node)
        if descendants_invariant_debug_enabled():
            _log_expansion_node(
                tree=tree,
                tree_expansion=tree_expansion,
                phase="after_register_child",
            )
            validate_descendants_tags(tree, phase="after_register_child")
    elif descendants_invariant_debug_enabled():
        _log_expansion_node(
            tree=tree,
            tree_expansion=tree_expansion,
            phase="reused_child",
        )

    tree_expansions.record(tree_expansion=tree_expansion)


def _log_expansion_node[NodeT: node.ITreeNode[Any]](
    *,
    tree: trees.Tree[NodeT],
    tree_expansion: TreeExpansion[NodeT],
    phase: str,
) -> None:
    """Log the child node touched by structural expansion diagnostics."""
    child_node = tree_expansion.child_node
    anemone_logger.info(
        "%s phase=%s status=event node_count=%s depth_count=%s "
        "mismatch_count=0 duplicate_state_tag_groups=0 node_id=%s "
        "node_tag=%r state_tag=%r node_depth=%s parent_node_id=%s "
        "creation_child_node=%s",
        DESCENDANTS_INVARIANT_LOG_PREFIX,
        phase,
        tree.descendants.get_count(),
        _descendants_depth_count(tree),
        child_node.id,
        child_node.tag,
        child_node.state.tag,
        child_node.tree_depth,
        tree_expansion.parent_node.id
        if tree_expansion.parent_node is not None
        else None,
        tree_expansion.creation_child_node,
    )


def _descendants_depth_count[NodeT: node.ITreeNode[Any]](
    tree: trees.Tree[NodeT],
) -> int:
    """Return the number of depth buckets currently held by descendants."""
    if tree.descendants.empty():
        return 0
    return len(tree.descendants.range())


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

    def record(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """Record one tree expansion under the appropriate bookkeeping bucket."""
        if tree_expansion.creation_child_node:
            self.record_creation(tree_expansion=tree_expansion)
        else:
            self.record_connection(tree_expansion=tree_expansion)

    def created_nodes(self) -> list[NodeT]:
        """Return the nodes that were newly created during this expansion wave."""
        return [
            tree_expansion.child_node
            for tree_expansion in self.expansions_with_node_creation
        ]

    def affected_child_nodes(self) -> list[NodeT]:
        """Return every child node touched during this expansion wave."""
        return [tree_expansion.child_node for tree_expansion in self]

    def add(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """Add a tree expansion to the collection.

        Args:
            tree_expansion: The tree expansion to add.

        """
        self.record(tree_expansion=tree_expansion)

    def record_creation(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """Record a tree expansion with a newly created child node."""
        self.expansions_with_node_creation.append(tree_expansion)

    def record_connection(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """Record a tree expansion that only connects to an existing child node."""
        self.expansions_without_node_creation.append(tree_expansion)

    def add_creation(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """Add a tree expansion with a newly created child node.

        Args:
            tree_expansion: The tree expansion to add.

        """
        self.record_creation(tree_expansion=tree_expansion)

    def add_connection(self, tree_expansion: TreeExpansion[NodeT]) -> None:
        """Add a tree expansion that only connects to an existing node.

        Args:
            tree_expansion: The tree expansion to add.

        """
        self.record_connection(tree_expansion=tree_expansion)

    def __str__(self) -> str:
        """Return a string summary of recorded expansions."""
        return (
            f"expansions_with_node_creation {self.expansions_with_node_creation} \n"
            f"expansions_without_node_creation{self.expansions_without_node_creation}"
        )
