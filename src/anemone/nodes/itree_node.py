"""Structural protocol shared by tree-node wrappers.

`ITreeNode` defines the navigation and branch-opening surface that generic tree
helpers are allowed to depend on. It intentionally stays structural: search
runtime details such as evaluations, exploration indices, and PV bookkeeping
belong on higher-level wrappers like `AlgorithmNode`, not on this protocol.
"""

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import Protocol, Self

from valanga import BranchKey, State, StateTag


class ITreeNode[StateT: State = State](Protocol):
    """Structural/navigation protocol for nodes stored in a tree."""

    @property
    def id(self) -> int:
        """Get the ID of the node.

        Returns:
            The ID of the node.

        """
        ...

    @property
    def state(self) -> StateT:
        """Get the state of the node.

        Returns:
            The state of the node.

        """
        ...

    @property
    def tree_depth(self) -> int:
        """Get the tree depth of the node.

        Returns:
            The tree depth of the node.

        """
        ...

    @property
    def branches_children(self) -> MutableMapping[BranchKey, Self | None]:
        """Materialize and return compatibility child-link storage.

        Core runtime code should prefer the explicit structural methods below.
        """
        ...

    def has_child_links(self) -> bool:
        """Return whether any structural child-link slot is stored."""
        ...

    def child_link_count(self) -> int:
        """Return the number of stored child-link slots."""
        ...

    def iter_child_links(self) -> Iterator[tuple[BranchKey, Self | None]]:
        """Iterate child-link slots without requiring materialized storage."""
        ...

    def iter_child_nodes(self) -> Iterator[Self]:
        """Iterate concrete child nodes."""
        ...

    def child_for_branch(self, branch: BranchKey) -> Self | None:
        """Return the concrete child for ``branch`` when linked."""
        ...

    def has_child_link_for_branch(self, branch: BranchKey) -> bool:
        """Return whether ``branch`` has a stored child-link slot."""
        ...

    def has_concrete_child_for_branch(self, branch: BranchKey) -> bool:
        """Return whether ``branch`` maps to a concrete child node."""
        ...

    def has_child_for_branch(self, branch: BranchKey) -> bool:
        """Compatibility alias for concrete-child semantics."""
        ...

    def set_child_for_branch(self, branch: BranchKey, child: Self | None) -> None:
        """Set one child-link slot."""
        ...

    def remove_child_link(self, branch: BranchKey) -> None:
        """Remove one child-link slot."""
        ...

    def clear_child_links(self) -> None:
        """Remove all child-link slots."""
        ...

    @property
    def parent_nodes(self) -> dict[Self, set[BranchKey]]:
        """Return the incoming parent-edge mapping for this node.

        Each key is a parent node. Each value is the set of distinct branch keys
        through which that parent reaches this node.
        """
        ...

    def parent_nodes_view(self) -> Mapping[Self, set[BranchKey]]:
        """Return a read-only mapping view over parent edges."""
        ...

    def iter_parent_items(self) -> Iterator[tuple[Self, set[BranchKey]]]:
        """Iterate parent-edge items without requiring a materialized dict."""
        ...

    def parent_count(self) -> int:
        """Return the number of distinct parent nodes."""
        ...

    def add_parent_link(self, parent_node: Self, branch: BranchKey) -> None:
        """Add one incoming parent edge using the compact storage helpers."""
        ...

    def add_parent(self, branch_key: BranchKey, new_parent_node: Self) -> None:
        """Add a parent node to the node.

        Args:
            new_parent_node: The parent node to add.
            branch_key: The branch key that led to the node from the new parent.

        """

    @property
    def all_branches_generated(self) -> bool:
        """Check if all branches have been generated.

        Returns:
            True if all branches have been generated, False otherwise.

        """
        ...

    @all_branches_generated.setter
    def all_branches_generated(self, value: bool) -> None:
        """Set the flag indicating that all branches have been generated."""

    @property
    def non_opened_branches(self) -> set[BranchKey]:
        """Materialize and return compatibility unopened-branch storage."""
        ...

    def has_unopened_branches(self) -> bool:
        """Return whether any unopened branches are stored."""
        ...

    def unopened_branch_count(self) -> int:
        """Return the stored unopened-branch count."""
        ...

    def iter_unopened_branches(self) -> Iterator[BranchKey]:
        """Iterate unopened branches without requiring materialized storage."""
        ...

    def contains_unopened_branch(self, branch: BranchKey) -> bool:
        """Return whether ``branch`` is currently stored as unopened."""
        ...

    def set_unopened_branches(self, branches: Iterable[BranchKey]) -> None:
        """Replace unopened branches."""
        ...

    def add_unopened_branch(self, branch: BranchKey) -> None:
        """Add one unopened branch."""
        ...

    def discard_unopened_branch(self, branch: BranchKey) -> None:
        """Discard one unopened branch."""
        ...

    def clear_unopened_branches(self) -> None:
        """Remove all unopened branches."""
        ...

    @property
    def tag(self) -> StateTag:
        """Get the fast tag representation of the node.

        Returns:
            The fast tag representation of the node as a string.

        """
        ...

    def is_over(self) -> bool:
        """Check if the state is terminal.

        Returns:
            True if the state is terminal, False otherwise.

        """
        ...
