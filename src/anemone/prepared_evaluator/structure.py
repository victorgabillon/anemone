"""Read-only native structural bindings over exclusively detached node ownership."""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Protocol, cast

from valanga import BranchKey, StateTag

from anemone.nodes.state_handles import StateHandle  # noqa: TC001
from anemone.nodes.tree_node import TreeNode


class ReadNodeStructure(Protocol):
    """Minimum ordered graph navigation consumed by evaluation and backup."""

    @property
    def id(self) -> int:
        """Return the stable native node id."""
        ...

    @property
    def tree_depth(self) -> int:
        """Return the native absolute depth."""
        ...

    @property
    def tag(self) -> StateTag:
        """Return the native state identity."""
        ...

    @property
    def state_handle(self) -> StateHandle[Any]:
        """Read an arena-owned state binding, never an owner handle."""
        ...

    @property
    def all_branches_generated(self) -> bool:
        """Return whether all structural branches have been generated."""
        ...

    def iter_unopened_branches(self) -> Iterator[BranchKey]:
        """Read unopened membership without exposing the backing set."""
        ...

    def iter_child_links(self) -> Iterator[tuple[BranchKey, Any]]:
        """Read ordered outgoing edges without exposing their mapping."""
        ...

    def iter_parent_items(self) -> Iterator[tuple[Any, set[BranchKey]]]:
        """Read ordered parents with independently owned branch sets."""
        ...


class DetachedStructure(TreeNode[Any, Any]):
    """Native read behavior with immutable backing and refused structural writes."""

    _sealed: bool = False

    @classmethod
    def from_node(cls, node: ReadNodeStructure) -> DetachedStructure:
        """Copy native structural bindings; all linked nodes belong to the arena."""
        result = cls(
            node.id,
            node.tree_depth,
            node.state_handle,
            None,
            node.all_branches_generated,
        )
        object.__setattr__(
            result,
            "branches_children_",
            MappingProxyType(dict(node.iter_child_links())),
        )
        object.__setattr__(
            result,
            "parent_nodes_",
            MappingProxyType({p: frozenset(b) for p, b in node.iter_parent_items()}),
        )
        object.__setattr__(
            result, "non_opened_branches_", frozenset(node.iter_unopened_branches())
        )
        object.__setattr__(result, "_sealed", True)
        return result

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse mutation after native reconstruction has finished."""
        if self._sealed:
            msg = "detached structure is read-only"
            raise TypeError(msg)
        object.__setattr__(self, name, value)

    @property
    def parent_nodes(self) -> dict[Any, set[BranchKey]]:
        """Return copies, never the frozen parent branch sets."""
        return dict(self.iter_parent_items())

    def iter_parent_items(self) -> Iterator[tuple[Any, set[BranchKey]]]:
        """Preserve incoming order while returning branch-set copies."""
        # from_node installs a MappingProxyType; Pylint misreads the generic cast.
        for parent, branches in cast(  # pylint: disable=no-member
            "Mapping[Any, frozenset[BranchKey]]", self.parent_nodes_
        ).items():
            yield parent, set(branches)

    def parent_count(self) -> int:
        """Return distinct parent count."""
        return len(cast("Mapping[Any, object]", self.parent_nodes_))

    def parent_nodes_view(self) -> Mapping[Any, set[BranchKey]]:
        """Return a read-only map whose values are independent copies."""
        return MappingProxyType(self.parent_nodes)

    @property
    def non_opened_branches(self) -> set[BranchKey]:
        """Return independent branch membership."""
        return set(self.iter_unopened_branches())

    @non_opened_branches.setter
    def non_opened_branches(self, value: set[BranchKey]) -> None:  # pylint: disable=unused-argument
        """Refuse replacement of immutable unopened membership."""
        msg = "detached structure is read-only"
        raise TypeError(msg)

    def mutable_child_links(self) -> dict[BranchKey, Any]:
        """Refuse opening capability rather than silently changing a copy."""
        msg = "detached structure cannot open branches"
        raise TypeError(msg)

    def mutable_unopened_branches(self) -> set[BranchKey]:
        """Refuse opening capability."""
        msg = "detached structure cannot open branches"
        raise TypeError(msg)
