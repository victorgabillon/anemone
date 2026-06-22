"""Tests for reusable one-ply opening expansion execution."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
)
from anemone.tree_manager import (
    BranchOpeningService,
    DuplicateBranchOpenError,
    OnePlyOpeningExpansionExecutor,
    OpeningExpansionBudget,
    TreeExpansions,
    TreeManager,
)


@dataclass(frozen=True)
class _State:
    tag: str


@dataclass
class _BranchKeys:
    keys: list[int]

    def get_all(self) -> list[int]:
        """Return configured legal keys."""
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
        """Record an incoming edge."""
        self.parent_nodes.setdefault(new_parent_node, set()).add(branch_key)

    def iter_child_links(self) -> Iterator[tuple[int, "_Node | None"]]:
        """Iterate structural child links."""
        return iter(self.branches_children.items())

    def child_for_branch(self, branch: int) -> "_Node | None":
        """Return one linked child."""
        return self.branches_children.get(branch)

    def set_child_for_branch(self, branch: int, child: "_Node | None") -> None:
        """Set one linked child."""
        self.branches_children[branch] = child

    def set_unopened_branches(self, branches: Iterable[int]) -> None:
        """Replace unopened branches."""
        self.non_opened_branches = set(branches)

    def discard_unopened_branch(self, branch: int) -> None:
        """Discard one unopened branch."""
        self.non_opened_branches.discard(branch)


class _NodeFactory:
    def create(
        self,
        *,
        state_handle: object,
        tree_depth: int,
        count: int,
        branch_from_parent: int,
        parent_node: _Node,
        modifications: object | None,
    ) -> _Node:
        """Create a fake node from a materialized state handle."""
        del branch_from_parent, parent_node, modifications
        return _Node(
            id=count,
            state=state_handle.get(),
            tree_depth=tree_depth,
        )


class _Descendants:
    def __init__(self) -> None:
        self.registered: list[_Node] = []

    def is_new_generation(self, tree_depth: int) -> bool:
        """Every test expansion creates a fresh node."""
        del tree_depth
        return True

    def contains_tag_at_depth(self, *, tree_depth: int, state_tag: str) -> bool:
        """Every test expansion creates a fresh node."""
        del tree_depth, state_tag
        return False

    def node_at(self, *, tree_depth: int, state_tag: str) -> _Node:
        """Reject existing-node lookup."""
        raise AssertionError((tree_depth, state_tag))

    def register_descendant(self, node: _Node) -> None:
        """Record one registered child."""
        self.registered.append(node)


class _Tree:
    def __init__(self, root: _Node) -> None:
        self.root_node = root
        self.tree_root_tree_depth = root.tree_depth
        self.nodes_count = 1
        self.branch_count = 0
        self.descendants = _Descendants()

    def node_depth(self, node: _Node) -> int:
        """Return depth relative to root."""
        return node.tree_depth - self.tree_root_tree_depth


class _Dynamics:
    def __init__(self, legal_by_tag: dict[str, list[int]]) -> None:
        self.legal_by_tag = legal_by_tag
        self.legal_action_calls = 0

    def legal_actions(self, state: _State) -> _BranchKeys:
        """Return legal actions while counting opening-status syncs."""
        self.legal_action_calls += 1
        return _BranchKeys(self.legal_by_tag.get(state.tag, []))

    def step(self, state: _State, action: int, *, depth: int) -> object:
        """Return a tiny transition object."""
        del depth
        return SimpleNamespace(
            next_state=_State(tag=f"{state.tag}:{action}"),
            modifications=None,
        )


def _instructions(node: _Node, branches: list[int]) -> OpeningInstructions[_Node]:
    batch: OpeningInstructions[_Node] = OpeningInstructions()
    for branch in branches:
        batch[(node.id, branch)] = OpeningInstruction(node_to_open=node, branch=branch)
    return batch


def _tree_manager(dynamics: _Dynamics) -> TreeManager[_Node]:
    return TreeManager(node_factory=_NodeFactory(), dynamics=dynamics)


def test_one_ply_executor_records_expansions_and_syncs_opening_status() -> None:
    """One-ply executor reproduces the structural batch expansion behavior."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0, 1]})
    manager = _tree_manager(dynamics)
    executor = OnePlyOpeningExpansionExecutor(
        branch_opening_service=BranchOpeningService(tree_manager=manager),
        dynamics=dynamics,
    )

    expansions = executor.expand(
        tree=tree,
        opening_instructions=_instructions(root, [0, 1]),
    )

    assert len(expansions.expansions_with_node_creation) == 2
    assert expansions.expansions_without_node_creation == []
    assert set(root.branches_children) == {0, 1}
    assert root.all_branches_generated
    assert root.non_opened_branches == set()
    assert tree.branch_count == 2
    assert tree.nodes_count == 3
    assert [node.id for node in tree.descendants.registered] == [1, 2]


def test_one_ply_executor_syncs_each_parent_once_per_batch() -> None:
    """Opening two branches from one parent performs one status sync."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0, 1]})
    manager = _tree_manager(dynamics)
    executor = OnePlyOpeningExpansionExecutor(
        branch_opening_service=BranchOpeningService(tree_manager=manager),
        dynamics=dynamics,
    )

    executor.expand(tree=tree, opening_instructions=_instructions(root, [0, 1]))

    assert dynamics.legal_action_calls == 1


def test_one_ply_executor_respects_materialized_branch_budget() -> None:
    """One-ply expansion stops before opening beyond the runtime branch budget."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0, 1, 2]})
    manager = _tree_manager(dynamics)
    executor = OnePlyOpeningExpansionExecutor(
        branch_opening_service=BranchOpeningService(tree_manager=manager),
        dynamics=dynamics,
    )
    budget = OpeningExpansionBudget(remaining_branch_openings=2)

    expansions = executor.expand(
        tree=tree,
        opening_instructions=_instructions(root, [0, 1, 2]),
        budget=budget,
    )

    assert len(expansions.expansions_with_node_creation) == 2
    assert set(root.branches_children) == {0, 1}
    assert 2 not in root.branches_children
    assert tree.branch_count == 2
    assert budget.remaining_branch_openings == 0
    assert root.non_opened_branches == {2}


def test_branch_opening_service_calls_callback_for_each_opened_branch() -> None:
    """The service callback fires at the actual branch-opening primitive."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0, 1]})
    manager = _tree_manager(dynamics)
    opened: list[tuple[int, int]] = []
    executor = OnePlyOpeningExpansionExecutor(
        branch_opening_service=BranchOpeningService(
            tree_manager=manager,
            on_branch_opened=lambda parent, branch: opened.append((parent.id, branch)),
        ),
        dynamics=dynamics,
    )

    executor.expand(tree=tree, opening_instructions=_instructions(root, [0, 1]))

    assert opened == [(0, 0), (0, 1)]


def test_duplicate_branch_open_does_not_call_branch_callback() -> None:
    """Duplicate-open precheck runs before callback mutation."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    root.branches_children[0] = _Node(id=10, state=_State("root:0"), tree_depth=1)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0]})
    manager = _tree_manager(dynamics)
    opened: list[int] = []
    service = BranchOpeningService(
        tree_manager=manager,
        on_branch_opened=lambda parent, branch: opened.append(branch),
    )

    with pytest.raises(DuplicateBranchOpenError):
        service.open_branch(
            tree=tree,
            parent_node=root,
            branch=0,
            tree_expansions=TreeExpansions(),
        )

    assert opened == []
