"""Focused tests for ``DepthIndexPropagator`` scheduling and recomputation."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

from anemone.indices.node_indices.index_data import MaxDepthDescendants
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.updates.depth_index_propagator import DepthIndexPropagator


@dataclass(eq=False)
class _FakeNode:
    """Tiny node stub with the fields the depth propagator needs."""

    name: str
    tree_depth: int
    branches_children: dict[int, "_FakeNode | None"] = field(default_factory=dict)
    parent_nodes: dict["_FakeNode", int] = field(default_factory=dict)
    exploration_index_data: MaxDepthDescendants[Any, Any] | object | None = None

    def __post_init__(self) -> None:
        if self.exploration_index_data is None:
            self.exploration_index_data = MaxDepthDescendants(
                tree_node=SimpleNamespace(id=self.name)
            )

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


def test_single_changed_child_recomputes_parent_depth_metadata() -> None:
    leaf = _FakeNode(name="leaf", tree_depth=2)
    parent = _FakeNode(name="parent", tree_depth=1)
    _connect(parent, branch=0, child=leaf)

    propagator = DepthIndexPropagator()
    affected = propagator.propagate_from_changed_nodes([_as_algorithm_node(leaf)])

    assert affected == {parent}
    assert isinstance(parent.exploration_index_data, MaxDepthDescendants)
    assert parent.exploration_index_data.max_depth_descendants == 1


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
        return True

    propagator = DepthIndexPropagator(recompute_node_depth_index=recompute)
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

    propagator = DepthIndexPropagator()
    affected = propagator.propagate_from_changed_nodes(
        [_as_algorithm_node(shared_child)]
    )

    assert affected == {parent_a, parent_b}
    assert (
        cast(
            "MaxDepthDescendants[Any, Any]", parent_a.exploration_index_data
        ).max_depth_descendants
        == 1
    )
    assert (
        cast(
            "MaxDepthDescendants[Any, Any]", parent_b.exploration_index_data
        ).max_depth_descendants
        == 1
    )


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

    propagator = DepthIndexPropagator(recompute_node_depth_index=recompute)
    affected = propagator.propagate_from_changed_nodes([_as_algorithm_node(leaf)])

    assert affected == {lower_parent, upper_parent, root}
    assert recompute_calls == ["lower_parent", "upper_parent", "root"]


def test_default_recompute_uses_full_current_child_snapshot() -> None:
    changed_child = _FakeNode(name="changed_child", tree_depth=2)
    sibling = _FakeNode(name="sibling", tree_depth=2)
    grandchild = _FakeNode(name="grandchild", tree_depth=3)
    parent = _FakeNode(name="parent", tree_depth=1)
    _connect(parent, branch=0, child=changed_child)
    _connect(parent, branch=1, child=sibling)
    _connect(changed_child, branch=3, child=grandchild)

    propagator = DepthIndexPropagator()
    affected = propagator.propagate_from_changed_nodes([_as_algorithm_node(grandchild)])

    assert affected == {changed_child, parent}
    assert (
        cast(
            "MaxDepthDescendants[Any, Any]", changed_child.exploration_index_data
        ).max_depth_descendants
        == 1
    )
    assert (
        cast(
            "MaxDepthDescendants[Any, Any]", parent.exploration_index_data
        ).max_depth_descendants
        == 2
    )
