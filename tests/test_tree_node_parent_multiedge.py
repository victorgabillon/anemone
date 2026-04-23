"""Focused tests for ``TreeNode`` incoming parent-edge storage."""

from dataclasses import dataclass
from typing import Any

import pytest
from valanga import Color

from anemone.node_factory.base import TreeNodeFactory
from anemone.nodes.state_handles import MaterializedStateHandle
from anemone.nodes.tree_node import TreeNode
from tests.fake_yaml_game import FakeYamlState


class _ConcreteFakeYamlState(FakeYamlState):
    """Concrete fake state suitable for direct ``TreeNode`` tests."""

    def pprint(self) -> str:
        """Return a tiny printable representation."""
        return str(self.node_id)


@dataclass(eq=False, slots=True)
class _ParentNode:
    """Minimal hashable parent object for incoming-edge tests."""

    id: int


def _make_state(node_id: int) -> _ConcreteFakeYamlState:
    """Create a simple fake state with no children."""
    return _ConcreteFakeYamlState(
        node_id=node_id,
        children_by_id={node_id: []},
        turn=Color.WHITE,
    )


def test_tree_node_factory_wraps_single_parent_branch_in_set() -> None:
    """Factory-created child nodes should store incoming branches in a set."""
    factory = TreeNodeFactory[Any, _ConcreteFakeYamlState]()
    parent = _ParentNode(id=1)
    child_state = _make_state(2)

    child = factory.create(
        state_handle=MaterializedStateHandle(state_=child_state),
        tree_depth=1,
        count=2,
        parent_node=parent,
        branch_from_parent=7,
        modifications=None,
    )

    assert child.parent_nodes == {parent: {7}}
    assert child.state is child_state


def test_tree_node_factory_create_from_state_wraps_materialized_state_handle() -> None:
    """Concrete-state convenience creation should still expose the same state."""
    factory = TreeNodeFactory[Any, _ConcreteFakeYamlState]()
    child_state = _make_state(2)

    child = factory.create_from_state(
        state=child_state,
        tree_depth=1,
        count=2,
        parent_node=None,
        branch_from_parent=None,
        modifications=None,
    )

    assert isinstance(child.state_handle, MaterializedStateHandle)
    assert child.state is child_state


def test_tree_node_add_parent_allows_distinct_branches_from_same_parent() -> None:
    """A child may be reached from one parent through multiple distinct branches."""
    parent = _ParentNode(id=1)
    child = TreeNode[Any, _ConcreteFakeYamlState](
        state_handle_=MaterializedStateHandle(state_=_make_state(2)),
        tree_depth_=1,
        id_=2,
        parent_nodes_={},
    )

    child.add_parent(branch_key=3, new_parent_node=parent)
    child.add_parent(branch_key=5, new_parent_node=parent)

    assert child.parent_nodes == {parent: {3, 5}}


def test_tree_node_add_parent_rejects_duplicate_parent_branch_edge() -> None:
    """Re-adding the same parent/branch pair should still assert."""
    parent = _ParentNode(id=1)
    child = TreeNode[Any, _ConcreteFakeYamlState](
        state_handle_=MaterializedStateHandle(state_=_make_state(2)),
        tree_depth_=1,
        id_=2,
        parent_nodes_={},
    )

    child.add_parent(branch_key=3, new_parent_node=parent)

    with pytest.raises(AssertionError, match="Duplicate parent edge"):
        child.add_parent(branch_key=3, new_parent_node=parent)
