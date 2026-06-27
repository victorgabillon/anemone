"""Generic structural node implementation shared by tree wrappers."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from valanga import BranchKey, State, StateTag

from .itree_node import ITreeNode
from .state_handles import StateHandle

# TODO: replace the any with a defaut value in ITReenode when availble in python; 3.13?


@dataclass(slots=True)
class _SingleParentLink[FamilyT: ITreeNode[Any]]:
    """Compact storage for one parent and its incoming branch keys."""

    parent_node: FamilyT
    branch_keys: set[BranchKey]


class _ParentNodesView[FamilyT: ITreeNode[Any]](Mapping[FamilyT, set[BranchKey]]):
    """Read-only mapping view over compact parent-link storage."""

    __slots__ = ("_node",)

    def __init__(self, node: "TreeNode[FamilyT, Any]") -> None:
        self._node = node

    def __getitem__(self, parent_node: FamilyT) -> set[BranchKey]:
        storage = cast("_ParentNodesStorage[FamilyT]", self._node.parent_nodes_)
        if storage is None:
            raise KeyError(parent_node)
        if isinstance(storage, _SingleParentLink):
            if storage.parent_node == parent_node:
                return storage.branch_keys
            raise KeyError(parent_node)
        return storage[parent_node]

    def __iter__(self) -> Iterator[FamilyT]:
        for parent_node, _branch_keys in self._node.iter_parent_items():
            yield parent_node

    def __len__(self) -> int:
        return self._node.parent_count()


type _ParentNodesStorage[FamilyT: ITreeNode[Any]] = (
    None | _SingleParentLink[FamilyT] | dict[FamilyT, set[BranchKey]]
)

type _ParentNodesInput[FamilyT: ITreeNode[Any]] = (
    None
    | tuple[FamilyT, BranchKey]
    | _SingleParentLink[FamilyT]
    | dict[FamilyT, set[BranchKey]]
)


@dataclass(slots=True)
class TreeNode[
    FamilyT: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
]:
    r"""Concrete structural implementation of ``ITreeNode``.

    ``TreeNode`` stores navigation and branch-opening bookkeeping only:
    state-handle ownership, tree depth, parent/child links, and unopened-branch
    tracking. Search/runtime layers should wrap it rather than grow more
    algorithm semantics here.

    Attributes:
        id\_ (int): The number to identify this node for easier debugging.
        tree_depth\_ (int): The depth of the node in the tree.
        state\_handle\_ (StateHandle[StateT]): The explicit handle owned by the
            node.
        parent_nodes\_ (None | tuple[ITreeNode, BranchKey] |
            dict[ITreeNode, set[BranchKey]]): Parent-link storage accepted at
            construction time and normalized to a compact runtime shape.
        all_branches_generated (bool): Whether no legal branch remains unopened.
        non_opened_branches\_ (set[BranchKey] | None): Lazy legal branches
            without concrete child links.
        branches_children\_ (dict[BranchKey, ITreeNode | None] | None): Lazy
            dictionary mapping branches to child nodes.
        tag (str): The fast tag representation of the state.

    Methods:
        id(): Returns the id of the node.
        state(): Resolves the concrete state through the handle.
        tree_depth(): Returns the depth of the node.
        branches_children(): Returns the dictionary mapping branches to child nodes.
        parent_nodes(): Returns the parent node mapping.
        is_root_node(): Checks if the node is a root node.
        add_parent(new_parent_node: ITreeNode): Adds a parent node to the current node.
        is_over(): Checks if the state is terminal.
        print_branches_children(): Prints the branches-children links of the node.

    """

    # id is a number to identify this node for easier debug
    id_: int

    # the tree depth of this node
    tree_depth_: int

    # the node holds a state handle.
    state_handle_: StateHandle[StateT]

    # Each parent can reach this node through multiple distinct branch keys.
    parent_nodes_: _ParentNodesInput[FamilyT]

    # Opening status is synchronized from legal actions and concrete child links:
    # all_branches_generated means no legal branch remains openable, and
    # non_opened_branches stores the currently openable legal branches.
    all_branches_generated: bool = False

    non_opened_branches_: set[BranchKey] | None = None

    # dictionary mapping branches to children nodes. Node is set to None if not created
    branches_children_: dict[BranchKey, FamilyT | None] | None = None

    def __post_init__(self) -> None:
        """Normalize constructor parent-link input to compact runtime storage."""
        self.parent_nodes_ = self._normalize_parent_nodes(self.parent_nodes_)

    @staticmethod
    def _normalize_parent_nodes(
        parent_nodes: _ParentNodesInput[FamilyT],
    ) -> _ParentNodesStorage[FamilyT]:
        """Convert compatibility parent-link input to compact internal storage."""
        if parent_nodes is None:
            return None
        if isinstance(parent_nodes, _SingleParentLink):
            branch_keys = set(parent_nodes.branch_keys)
            if not branch_keys:
                return None
            return _SingleParentLink(
                parent_node=parent_nodes.parent_node,
                branch_keys=branch_keys,
            )
        if isinstance(parent_nodes, tuple):
            parent_node, branch_key = parent_nodes
            return _SingleParentLink(parent_node=parent_node, branch_keys={branch_key})

        normalized = {
            parent_node: set(branch_keys)
            for parent_node, branch_keys in parent_nodes.items()
            if branch_keys
        }
        if not normalized:
            return None
        if len(normalized) == 1:
            parent_node, branch_keys = next(iter(normalized.items()))
            return _SingleParentLink(parent_node=parent_node, branch_keys=branch_keys)
        return normalized

    @property
    def tag(self) -> StateTag:
        """Return the fast tag representation of the resolved state.

        Returns:
            StateTag: The fast tag representation of the resolved state.

        """
        return self.state.tag

    @property
    def id(self) -> int:
        """Returns the ID of the tree node.

        Returns:
            int: The ID of the tree node.

        """
        return self.id_

    @property
    def state(self) -> StateT:
        """Resolve and return the concrete state through this node's handle.

        Returns:
            State: The concrete state resolved through the handle.

        """
        return self.state_handle_.get()

    @property
    def state_handle(self) -> StateHandle[StateT]:
        """Return the explicit state handle owned by this node."""
        return self.state_handle_

    @property
    def tree_depth(self) -> int:
        """Returns the tree depth of this node.

        Returns:
            int: The tree depth of this node.

        """
        return self.tree_depth_

    @property
    def branches_children(self) -> dict[BranchKey, FamilyT | None]:
        """Materialize and return the compatibility child-link dictionary.

        Core runtime code should prefer the explicit non-materializing helpers
        such as ``iter_child_links`` and ``child_for_branch``.
        """
        return self.mutable_child_links()

    @branches_children.setter
    def branches_children(self, value: dict[BranchKey, FamilyT | None]) -> None:
        """Set compatibility child links, preserving lazy empty storage."""
        self.branches_children_ = dict(value) or None

    def has_child_links(self) -> bool:
        """Return whether any structural child-link slot is stored."""
        return bool(self.branches_children_)

    def child_link_count(self) -> int:
        """Return the number of stored child-link slots without allocating."""
        if self.branches_children_ is None:
            return 0
        return len(self.branches_children_)

    def iter_child_links(self) -> Iterator[tuple[BranchKey, FamilyT | None]]:
        """Iterate child-link slots without materializing empty storage."""
        if self.branches_children_ is None:
            return
        yield from self.branches_children_.items()

    def iter_child_nodes(self) -> Iterator[FamilyT]:
        """Iterate concrete non-``None`` child nodes without allocating."""
        if self.branches_children_ is None:
            return
        for child in self.branches_children_.values():
            if child is not None:
                yield child

    def child_for_branch(self, branch: BranchKey) -> FamilyT | None:
        """Return the child linked to ``branch``, if any, without allocating."""
        if self.branches_children_ is None:
            return None
        return self.branches_children_.get(branch)

    def has_child_link_for_branch(self, branch: BranchKey) -> bool:
        """Return whether ``branch`` has a stored child-link slot, even if ``None``."""
        return self.branches_children_ is not None and branch in self.branches_children_

    def has_concrete_child_for_branch(self, branch: BranchKey) -> bool:
        """Return whether ``branch`` maps to a concrete non-``None`` child."""
        return self.child_for_branch(branch) is not None

    def has_child_for_branch(self, branch: BranchKey) -> bool:
        """Compatibility alias for concrete-child semantics.

        Prefer ``has_child_link_for_branch`` when a stored ``branch -> None``
        slot should count, and ``has_concrete_child_for_branch`` when only a
        real child node should count.
        """
        return self.has_concrete_child_for_branch(branch)

    def mutable_child_links(self) -> dict[BranchKey, FamilyT | None]:
        """Materialize and return the mutable child-link dictionary."""
        if self.branches_children_ is None:
            self.branches_children_ = {}
        return self.branches_children_

    def set_child_for_branch(self, branch: BranchKey, child: FamilyT | None) -> None:
        """Set one child-link slot, materializing storage only for mutation."""
        self.mutable_child_links()[branch] = child

    def remove_child_link(self, branch: BranchKey) -> None:
        """Remove one child-link slot and release empty storage."""
        if self.branches_children_ is None:
            return
        self.branches_children_.pop(branch, None)
        if not self.branches_children_:
            self.branches_children_ = None

    def clear_child_links(self) -> None:
        """Remove all child links and release the backing dictionary."""
        self.branches_children_ = None

    @property
    def non_opened_branches(self) -> set[BranchKey]:
        """Materialize and return the compatibility unopened-branch set.

        Core runtime code should prefer explicit helpers such as
        ``iter_unopened_branches`` and ``set_unopened_branches``.
        """
        return self.mutable_unopened_branches()

    @non_opened_branches.setter
    def non_opened_branches(self, value: set[BranchKey]) -> None:
        """Set compatibility unopened branches, preserving lazy empty storage."""
        self.non_opened_branches_ = set(value) or None

    def has_unopened_branches(self) -> bool:
        """Return whether any unopened branches are stored."""
        return bool(self.non_opened_branches_)

    def unopened_branch_count(self) -> int:
        """Return the stored unopened-branch count without allocating."""
        if self.non_opened_branches_ is None:
            return 0
        return len(self.non_opened_branches_)

    def iter_unopened_branches(self) -> Iterator[BranchKey]:
        """Iterate unopened branches without materializing empty storage."""
        if self.non_opened_branches_ is None:
            return
        yield from self.non_opened_branches_

    def contains_unopened_branch(self, branch: BranchKey) -> bool:
        """Return whether ``branch`` is currently stored as unopened."""
        if self.non_opened_branches_ is None:
            return False
        return branch in self.non_opened_branches_

    def mutable_unopened_branches(self) -> set[BranchKey]:
        """Materialize and return the mutable unopened-branch set."""
        if self.non_opened_branches_ is None:
            self.non_opened_branches_ = set()
        return self.non_opened_branches_

    def set_unopened_branches(self, branches: Iterable[BranchKey]) -> None:
        """Replace unopened branches, storing ``None`` for empty input."""
        self.non_opened_branches_ = set(branches) or None

    def add_unopened_branch(self, branch: BranchKey) -> None:
        """Add one unopened branch, materializing storage only for mutation."""
        self.mutable_unopened_branches().add(branch)

    def discard_unopened_branch(self, branch: BranchKey) -> None:
        """Discard one unopened branch without materializing empty storage."""
        if self.non_opened_branches_ is None:
            return
        self.non_opened_branches_.discard(branch)
        if not self.non_opened_branches_:
            self.non_opened_branches_ = None

    def clear_unopened_branches(self) -> None:
        """Remove all unopened branches and release the backing set."""
        self.non_opened_branches_ = None

    @property
    def parent_nodes(self) -> dict[FamilyT, set[BranchKey]]:
        """Return the compatibility incoming parent-edge mapping for this node.

        Each key is a parent node. Each value is the set of distinct branch keys
        through which that parent reaches this node.
        """
        storage = cast("_ParentNodesStorage[FamilyT]", self.parent_nodes_)
        if storage is None:
            return {}
        if isinstance(storage, _SingleParentLink):
            return {storage.parent_node: storage.branch_keys}
        return storage

    def parent_nodes_view(self) -> Mapping[FamilyT, set[BranchKey]]:
        """Return a read-only mapping view over incoming parent edges."""
        return _ParentNodesView(self)

    def iter_parent_items(self) -> Iterator[tuple[FamilyT, set[BranchKey]]]:
        """Iterate parent-edge items without materializing zero/one-parent dicts."""
        storage = cast("_ParentNodesStorage[FamilyT]", self.parent_nodes_)
        if storage is None:
            return
        if isinstance(storage, _SingleParentLink):
            yield storage.parent_node, storage.branch_keys
            return
        yield from storage.items()

    def parent_count(self) -> int:
        """Return the number of distinct parent nodes without allocating."""
        storage = cast("_ParentNodesStorage[FamilyT]", self.parent_nodes_)
        if storage is None:
            return 0
        if isinstance(storage, _SingleParentLink):
            return 1
        return len(storage)

    def add_parent_link(self, parent_node: FamilyT, branch: BranchKey) -> None:
        """Add one incoming parent edge using compact zero/one/many storage."""
        storage = cast("_ParentNodesStorage[FamilyT]", self.parent_nodes_)
        if storage is None:
            self.parent_nodes_ = _SingleParentLink(
                parent_node=parent_node,
                branch_keys={branch},
            )
            return

        if isinstance(storage, _SingleParentLink):
            if storage.parent_node == parent_node:
                assert branch not in storage.branch_keys, (
                    f"Duplicate parent edge for child {self.id} from parent "
                    f"{parent_node.id} via branch {branch!r}"
                )
                storage.branch_keys.add(branch)
                return
            self.parent_nodes_ = {
                storage.parent_node: storage.branch_keys,
                parent_node: {branch},
            }
            return

        branch_keys = storage.setdefault(parent_node, set())
        assert branch not in branch_keys, (
            f"Duplicate parent edge for child {self.id} from parent "
            f"{parent_node.id} via branch {branch!r}"
        )
        branch_keys.add(branch)

    def is_root_node(self) -> bool:
        """Check if the current node is a root node.

        Returns:
            bool: True if the node is a root node, False otherwise.

        """
        return self.parent_count() == 0

    def add_parent(self, branch_key: BranchKey, new_parent_node: FamilyT) -> None:
        """Add a new parent node to the current node.

        Args:
            branch_key (BranchKey): The branch key that led to the node from the new parent node.
            new_parent_node (ITreeNode): The new parent node to be added.

        Raises:
            AssertionError: If the parent/branch edge already exists.

        Returns:
            None

        """
        self.add_parent_link(parent_node=new_parent_node, branch=branch_key)

    def is_over(self) -> bool:
        """Check if the resolved state is terminal.

        Returns:
            bool: True if the resolved state is terminal, False otherwise.

        """
        return self.state.is_game_over()

    def print_branches_children(self) -> None:
        """Print the branches-children link of the node.

        This method prints the branches-children link of the node, showing the branch and the ID of the child node.
        If a child node is None, it will be displayed as 'None'.

        Returns:
            None

        """
        print(
            "here are the ",
            self.child_link_count(),
            " branches-children link of node",
            self.id,
            ": ",
            end=" ",
        )
        for branch, child in self.iter_child_links():
            if child is None:
                print(branch, child, end=" ")
            else:
                print(branch, child.id, end=" ")
        print(" ")
