"""Tests for deterministic materialized rollout expansion."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
)
from anemone.rollouts import (
    FirstOpenableActionPolicy,
    NoRolloutPolicy,
    RolloutOpeningExpansionExecutor,
    RolloutStopReason,
)
from anemone.tree_manager import BranchOpeningService, TreeManager

if TYPE_CHECKING:
    from collections.abc import Callable


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
    branches_children: dict[int, _Node | None] = field(default_factory=dict)
    all_branches_generated: bool = False
    non_opened_branches: set[int] = field(default_factory=set)
    parent_nodes: dict[_Node, set[int]] = field(default_factory=dict)
    terminal: bool = False

    def add_parent(self, branch_key: int, new_parent_node: _Node) -> None:
        """Record an incoming edge."""
        self.parent_nodes.setdefault(new_parent_node, set()).add(branch_key)

    def is_over(self) -> bool:
        """Return whether this fake node is terminal."""
        return self.terminal


@dataclass
class _NodeFactory:
    preopened_branches_by_tag: dict[str, set[int]] = field(default_factory=dict)

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
        state = state_handle.get()
        node = _Node(id=count, state=state, tree_depth=tree_depth)
        for branch in self.preopened_branches_by_tag.get(state.tag, set()):
            node.branches_children[branch] = _Node(
                id=1000 + branch,
                state=_State(f"{state.tag}:preopened:{branch}"),
                tree_depth=tree_depth + 1,
            )
        return node


class _Descendants:
    def __init__(self) -> None:
        self._nodes_by_depth_tag: dict[tuple[int, str], _Node] = {}
        self.registered: list[_Node] = []

    def is_new_generation(self, tree_depth: int) -> bool:
        """Return whether no node exists yet at ``tree_depth``."""
        return not any(depth == tree_depth for depth, _tag in self._nodes_by_depth_tag)

    def contains_tag_at_depth(self, *, tree_depth: int, state_tag: str) -> bool:
        """Return whether the tag is already known at depth."""
        return (tree_depth, state_tag) in self._nodes_by_depth_tag

    def node_at(self, *, tree_depth: int, state_tag: str) -> _Node:
        """Return an existing fake node."""
        return self._nodes_by_depth_tag[(tree_depth, state_tag)]

    def register_descendant(self, node: _Node) -> None:
        """Register a created fake node."""
        self.registered.append(node)
        self.register_existing(node)

    def register_existing(self, node: _Node) -> None:
        """Register an existing fake node without counting it as newly created."""
        self._nodes_by_depth_tag[(node.tree_depth, node.state.tag)] = node


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

    def legal_actions(self, state: _State) -> _BranchKeys:
        """Return legal actions for a fake state."""
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


def _executor(
    *,
    dynamics: _Dynamics,
    max_extra_steps: int,
    policy: object | None = None,
    opened: list[tuple[int, int]] | None = None,
    node_factory: _NodeFactory | None = None,
) -> RolloutOpeningExpansionExecutor[_Node]:
    manager = TreeManager(
        node_factory=node_factory or _NodeFactory(),
        dynamics=dynamics,
    )
    callback: Callable[[_Node, int], None] | None = None
    if opened is not None:

        def callback(parent: _Node, branch: int) -> None:
            opened.append((parent.id, branch))

    return RolloutOpeningExpansionExecutor(
        branch_opening_service=BranchOpeningService(
            tree_manager=manager,
            on_branch_opened=callback,
        ),
        dynamics=dynamics,
        rollout_policy=policy or FirstOpenableActionPolicy(),
        max_extra_steps=max_extra_steps,
    )


def test_max_extra_steps_zero_matches_one_ply() -> None:
    """Zero extra rollout steps opens only the initial instruction edge."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0], "root:0": [1]})
    executor = _executor(dynamics=dynamics, max_extra_steps=0)

    expansions = executor.expand(
        tree=tree, opening_instructions=_instructions(root, [0])
    )

    child = root.branches_children[0]
    assert child is not None
    assert child.branches_children == {}
    assert len(expansions.expansions_with_node_creation) == 1
    assert expansions.expansions_without_node_creation == []
    assert tree.nodes_count == 2
    assert root.all_branches_generated
    assert executor.last_report is not None
    assert executor.last_report.extra_edge_count == 0
    assert (
        executor.last_report.stop_reason_counts[RolloutStopReason.MAX_EXTRA_STEPS.value]
        == 1
    )


def test_deterministic_rollout_creates_chain() -> None:
    """First-openable rollout materializes a deterministic continuation chain."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics(
        {
            "root": [0],
            "root:0": [1],
            "root:0:1": [2],
            "root:0:1:2": [],
        }
    )
    executor = _executor(dynamics=dynamics, max_extra_steps=2)

    expansions = executor.expand(
        tree=tree, opening_instructions=_instructions(root, [0])
    )

    first = root.branches_children[0]
    assert first is not None
    second = first.branches_children[1]
    assert second is not None
    third = second.branches_children[2]
    assert third is not None
    assert len(expansions.expansions_with_node_creation) == 3
    assert tree.nodes_count == 4
    assert tree.branch_count == 3
    assert root.all_branches_generated
    assert first.all_branches_generated
    assert second.all_branches_generated
    assert executor.last_report is not None
    assert executor.last_report.extra_edge_count == 2
    assert executor.last_report.total_edge_count == 3
    assert executor.last_report.max_extra_depth_reached == 2


def test_no_rollout_policy_stops_after_initial_edge() -> None:
    """NoRolloutPolicy opts out after opening the selected edge."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0], "root:0": [1]})
    executor = _executor(
        dynamics=dynamics,
        max_extra_steps=3,
        policy=NoRolloutPolicy(),
    )

    expansions = executor.expand(
        tree=tree, opening_instructions=_instructions(root, [0])
    )

    child = root.branches_children[0]
    assert child is not None
    assert child.branches_children == {}
    assert len(expansions.expansions_with_node_creation) == 1
    assert executor.last_report is not None
    assert (
        executor.last_report.stop_reason_counts[RolloutStopReason.POLICY_STOP.value]
        == 1
    )


def test_no_openable_actions_stops_after_initial_edge() -> None:
    """A child with no openable branches stops rollout."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0], "root:0": []})
    executor = _executor(dynamics=dynamics, max_extra_steps=3)

    executor.expand(tree=tree, opening_instructions=_instructions(root, [0]))

    assert executor.last_report is not None
    assert (
        executor.last_report.stop_reason_counts[
            RolloutStopReason.NO_OPENABLE_ACTIONS.value
        ]
        == 1
    )


def test_existing_initial_child_stops_rollout() -> None:
    """Initial existing-node connection is recorded and not rolled through."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    existing = _Node(id=99, state=_State("root:0"), tree_depth=1)
    tree.descendants.register_existing(existing)
    dynamics = _Dynamics({"root": [0], "root:0": [1]})
    executor = _executor(dynamics=dynamics, max_extra_steps=3)

    expansions = executor.expand(
        tree=tree, opening_instructions=_instructions(root, [0])
    )

    assert root.branches_children[0] is existing
    assert existing.branches_children == {}
    assert expansions.expansions_with_node_creation == []
    assert len(expansions.expansions_without_node_creation) == 1
    assert executor.last_report is not None
    assert executor.last_report.existing_node_stop_count == 1


def test_existing_node_reached_during_rollout_stops_after_recording_edge() -> None:
    """Existing-node rollout edge is recorded, then rollout stops."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    existing = _Node(id=99, state=_State("root:0:1"), tree_depth=2)
    tree.descendants.register_existing(existing)
    dynamics = _Dynamics({"root": [0], "root:0": [1], "root:0:1": [2]})
    executor = _executor(dynamics=dynamics, max_extra_steps=3)

    expansions = executor.expand(
        tree=tree, opening_instructions=_instructions(root, [0])
    )

    child = root.branches_children[0]
    assert child is not None
    assert child.branches_children[1] is existing
    assert existing.branches_children == {}
    assert len(expansions.expansions_with_node_creation) == 1
    assert len(expansions.expansions_without_node_creation) == 1
    assert executor.last_report is not None
    assert executor.last_report.extra_edge_count == 1
    assert executor.last_report.existing_node_stop_count == 1


def test_rollout_edges_call_branch_opened_callback() -> None:
    """Every materialized rollout edge goes through BranchOpeningService."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0], "root:0": [1], "root:0:1": [2]})
    opened: list[tuple[int, int]] = []
    executor = _executor(dynamics=dynamics, max_extra_steps=2, opened=opened)

    executor.expand(tree=tree, opening_instructions=_instructions(root, [0]))

    assert opened == [(0, 0), (1, 1), (2, 2)]


def test_rollout_uses_openable_actions_not_all_legal_actions() -> None:
    """Rollout policy receives only legal branches that are not already opened."""
    root = _Node(id=0, state=_State("root"), tree_depth=0)
    tree = _Tree(root)
    dynamics = _Dynamics({"root": [0], "root:0": [0, 1, 2], "root:0:1": []})
    executor = _executor(
        dynamics=dynamics,
        max_extra_steps=1,
        node_factory=_NodeFactory(preopened_branches_by_tag={"root:0": {0}}),
    )

    executor.expand(tree=tree, opening_instructions=_instructions(root, [0]))

    child = root.branches_children[0]
    assert child is not None
    assert set(child.branches_children) == {0, 1}
    assert child.branches_children[0] is not None
    assert child.branches_children[1] is not None
    assert 2 in child.non_opened_branches
