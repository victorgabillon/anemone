"""Focused tests for ``ValuePropagator`` scheduling semantics."""

from dataclasses import dataclass, field
from typing import Any, cast

from anemone.backup_policies.types import BackupResult
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.updates.value_propagator import ValuePropagator


@dataclass(eq=False)
class _FakeEvaluation:
    """Minimal backup surface for testing propagation orchestration."""

    results: list[BackupResult] = field(default_factory=list)
    calls: list[tuple[set[int], set[int]]] = field(default_factory=list)

    def backup_from_children(
        self,
        branches_with_updated_value: set[int],
        branches_with_updated_best_branch_seq: set[int],
    ) -> BackupResult:
        """Record one backup call and return the next configured result."""
        self.calls.append(
            (
                set(branches_with_updated_value),
                set(branches_with_updated_best_branch_seq),
            )
        )
        if self.results:
            return self.results.pop(0)
        return BackupResult(value_changed=False, pv_changed=False, over_changed=False)


@dataclass(eq=False)
class _FakeNode:
    """Tiny node stub with just the fields the propagator needs."""

    name: str
    tree_depth: int
    branches_children: dict[int, "_FakeNode | None"] = field(default_factory=dict)
    parent_nodes: dict["_FakeNode", int] = field(default_factory=dict)
    tree_evaluation: _FakeEvaluation = field(default_factory=_FakeEvaluation)

    def __repr__(self) -> str:
        """Return a stable name to make assertion failures readable."""
        return self.name


def _connect(parent: _FakeNode, branch: int, child: _FakeNode) -> None:
    """Connect two fake nodes through one branch."""
    parent.branches_children[branch] = child
    child.parent_nodes[parent] = branch


def _as_algorithm_node(node: _FakeNode) -> AlgorithmNode[Any]:
    """Cast a fake node to the production node type used by the propagator."""
    return cast("AlgorithmNode[Any]", node)


def test_single_changed_child_recomputes_parent() -> None:
    leaf = _FakeNode(name="leaf", tree_depth=2)
    parent = _FakeNode(name="parent", tree_depth=1)
    _connect(parent, branch=0, child=leaf)

    recompute_calls: list[str] = []

    def recompute(node: AlgorithmNode[Any]) -> bool:
        recompute_calls.append(cast("_FakeNode", node).name)
        return False

    propagator = ValuePropagator(recompute_node_value=recompute)
    affected = propagator.propagate_from_changed_nodes([_as_algorithm_node(leaf)])

    assert affected == {parent}
    assert recompute_calls == ["parent"]


def test_two_changed_siblings_recompute_parent_once_before_grandparent() -> None:
    left = _FakeNode(name="left", tree_depth=2)
    right = _FakeNode(name="right", tree_depth=2)
    parent = _FakeNode(name="parent", tree_depth=1)
    grandparent = _FakeNode(name="grandparent", tree_depth=0)
    _connect(parent, branch=0, child=left)
    _connect(parent, branch=1, child=right)
    _connect(grandparent, branch=7, child=parent)

    recompute_calls: list[str] = []

    def recompute(node: AlgorithmNode[Any]) -> bool:
        fake_node = cast("_FakeNode", node)
        recompute_calls.append(fake_node.name)
        return fake_node is parent

    propagator = ValuePropagator(recompute_node_value=recompute)
    affected = propagator.propagate_from_changed_nodes(
        [_as_algorithm_node(left), _as_algorithm_node(right)]
    )

    assert affected == {parent, grandparent}
    assert recompute_calls == ["parent", "grandparent"]


def test_changed_child_with_two_parents_schedules_both_parents() -> None:
    shared_child = _FakeNode(name="shared_child", tree_depth=2)
    parent_a = _FakeNode(name="parent_a", tree_depth=1)
    parent_b = _FakeNode(name="parent_b", tree_depth=1)
    _connect(parent_a, branch=0, child=shared_child)
    _connect(parent_b, branch=1, child=shared_child)

    recompute_calls: list[str] = []

    def recompute(node: AlgorithmNode[Any]) -> bool:
        recompute_calls.append(cast("_FakeNode", node).name)
        return False

    propagator = ValuePropagator(recompute_node_value=recompute)
    affected = propagator.propagate_from_changed_nodes(
        [_as_algorithm_node(shared_child)]
    )

    assert affected == {parent_a, parent_b}
    assert set(recompute_calls) == {"parent_a", "parent_b"}
    assert len(recompute_calls) == 2


def test_depth_waves_process_deeper_dirty_nodes_before_shallower_ancestors() -> None:
    leaf = _FakeNode(name="leaf", tree_depth=3)
    lower_parent = _FakeNode(name="lower_parent", tree_depth=2)
    upper_parent = _FakeNode(name="upper_parent", tree_depth=1)
    root = _FakeNode(name="root", tree_depth=0)
    _connect(lower_parent, branch=0, child=leaf)
    _connect(upper_parent, branch=1, child=lower_parent)
    _connect(root, branch=2, child=upper_parent)

    recompute_calls: list[str] = []

    def recompute(node: AlgorithmNode[Any]) -> bool:
        fake_node = cast("_FakeNode", node)
        recompute_calls.append(fake_node.name)
        return fake_node in {lower_parent, upper_parent}

    propagator = ValuePropagator(recompute_node_value=recompute)
    affected = propagator.propagate_from_changed_nodes([_as_algorithm_node(leaf)])

    assert affected == {lower_parent, upper_parent, root}
    assert recompute_calls == ["lower_parent", "upper_parent", "root"]


def test_default_recompute_uses_full_current_child_snapshot() -> None:
    changed_child = _FakeNode(name="changed_child", tree_depth=2)
    sibling = _FakeNode(name="sibling", tree_depth=2)
    parent = _FakeNode(name="parent", tree_depth=1)
    parent.tree_evaluation = _FakeEvaluation(
        results=[BackupResult(value_changed=True, pv_changed=False, over_changed=False)]
    )
    _connect(parent, branch=0, child=changed_child)
    _connect(parent, branch=1, child=sibling)

    propagator = ValuePropagator()
    affected = propagator.propagate_from_changed_nodes(
        [_as_algorithm_node(changed_child)]
    )

    assert affected == {parent}
    assert parent.tree_evaluation.calls == [({0, 1}, {0, 1})]
