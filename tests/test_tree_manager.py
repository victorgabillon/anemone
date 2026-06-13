"""Tests for structural tree-manager opening safety."""

from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace

import pytest

from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
    OpeningInstructor,
    OpeningType,
)
from anemone.tree_manager import DuplicateBranchOpenError, TreeManager


@dataclass(frozen=True)
class _State:
    tag: str


@dataclass
class _BranchKeys:
    keys: list[int]

    def get_all(self) -> list[int]:
        """Return legal keys in configured order."""
        return self.keys


@dataclass(eq=False)
class _Node:
    id: int
    state: _State
    tree_depth: int
    branches_children: dict[int, "_Node | None"] = field(default_factory=dict)
    all_branches_generated: bool = False
    non_opened_branches: set[int] = field(default_factory=set)
    parent_nodes: dict["_Node", set[int]] = field(default_factory=dict)

    def add_parent(self, branch_key: int, new_parent_node: "_Node") -> None:
        """Record an incoming edge for existing-node connections."""
        self.parent_nodes.setdefault(new_parent_node, set()).add(branch_key)


class _StateHandle:
    def __init__(self, state: _State) -> None:
        self._state = state

    def get(self) -> _State:
        """Return the materialized state."""
        return self._state


class _NodeFactory:
    def create(
        self,
        *,
        state_handle: _StateHandle,
        tree_depth: int,
        count: int,
        branch_from_parent: int,
        parent_node: _Node,
        modifications: object | None,
    ) -> _Node:
        """Create a fake child node."""
        del branch_from_parent, parent_node, modifications
        return _Node(id=count, state=state_handle.get(), tree_depth=tree_depth)


class _Descendants:
    def __init__(self) -> None:
        self.registered: list[_Node] = []

    def is_new_generation(self, tree_depth: int) -> bool:
        """Return True so every expansion creates a child node."""
        del tree_depth
        return True

    def contains_tag_at_depth(self, *, tree_depth: int, state_tag: str) -> bool:
        """Return False so every expansion creates a child node."""
        del tree_depth, state_tag
        return False

    def node_at(self, *, tree_depth: int, state_tag: str) -> _Node:
        """Reject existing descendant lookups."""
        raise AssertionError((tree_depth, state_tag))

    def register_descendant(self, node: _Node) -> None:
        """Record one created node."""
        self.registered.append(node)


class _Tree:
    def __init__(self, root: _Node) -> None:
        self.root_node = root
        self.tree_root_tree_depth = root.tree_depth
        self.nodes_count = 1
        self.branch_count = 0
        self.descendants = _Descendants()

    def node_depth(self, node: _Node) -> int:
        """Return depth relative to the root."""
        return node.tree_depth - self.tree_root_tree_depth


class _Dynamics:
    def __init__(self, legal_by_tag: dict[str, list[int]]) -> None:
        self.legal_by_tag = legal_by_tag

    def legal_actions(self, state: _State) -> _BranchKeys:
        """Return legal keys for ``state``."""
        return _BranchKeys(self.legal_by_tag.get(state.tag, []))

    def step(self, state: _State, action: int, *, depth: int) -> object:
        """Return a tiny transition object."""
        del depth
        return SimpleNamespace(
            next_state=_State(tag=f"{state.tag}:{action}"),
            modifications=None,
        )

    def action_name(self, state: _State, action: int) -> str:
        """Return a printable branch name."""
        del state
        return str(action)


def _instructions(node: _Node, branches: list[int]) -> OpeningInstructions[_Node]:
    batch: OpeningInstructions[_Node] = OpeningInstructions()
    for branch in branches:
        batch[(node.id, branch)] = OpeningInstruction(node_to_open=node, branch=branch)
    return batch


def test_limited_expansion_keeps_node_partially_open_until_remaining_branches_open() -> (
    None
):
    """Dropped instructions must not mark a node fully generated."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    manager = TreeManager(
        node_factory=_NodeFactory(),
        dynamics=_Dynamics({"root": [0, 1, 2]}),
    )

    manager.expand_instructions(
        tree=tree, opening_instructions=_instructions(root, [0])
    )

    assert set(root.branches_children) == {0}
    assert not root.all_branches_generated
    assert root.non_opened_branches == {1, 2}

    manager.expand_instructions(
        tree=tree,
        opening_instructions=_instructions(root, [1, 2]),
    )

    assert set(root.branches_children) == {0, 1, 2}
    assert root.all_branches_generated
    assert root.non_opened_branches == set()


def test_tree_manager_rejects_duplicate_branch_opening() -> None:
    """Opening a branch with an existing child is a structural error."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    manager = TreeManager(
        node_factory=_NodeFactory(),
        dynamics=_Dynamics({"root": [1]}),
    )
    manager.expand_instructions(
        tree=tree, opening_instructions=_instructions(root, [1])
    )

    with pytest.raises(DuplicateBranchOpenError, match="branch already has child"):
        manager.expand_instructions(
            tree=tree,
            opening_instructions=_instructions(root, [1]),
        )


def test_all_children_completes_partially_opened_node_without_duplicates() -> None:
    """The instructor/tree-manager path opens only missing branches and then syncs."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    existing_child = _Node(id=99, state=_State("root:1"), tree_depth=1)
    root.branches_children[1] = existing_child
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0, 1, 2]})
    instructor = OpeningInstructor(
        opening_type=OpeningType.ALL_CHILDREN,
        random_generator=Random(0),
        dynamics=dynamics,
    )
    manager = TreeManager(node_factory=_NodeFactory(), dynamics=dynamics)

    branches_to_open = instructor.all_branches_to_open(node_to_open=root)
    assert branches_to_open == (0, 2)

    manager.expand_instructions(
        tree=tree,
        opening_instructions=_instructions(root, list(branches_to_open)),
    )

    assert set(root.branches_children) == {0, 1, 2}
    assert root.branches_children[1] is existing_child
    assert root.all_branches_generated
    assert root.non_opened_branches == set()
