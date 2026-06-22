"""Tests for lazy ``TreeNode`` branch container storage."""

from types import SimpleNamespace
from typing import Any

from valanga import Color

from anemone.nodes.opening_status import sync_opening_status
from anemone.nodes.state_handles import MaterializedStateHandle
from anemone.nodes.tree_node import TreeNode
from tests.fake_yaml_game import FakeYamlState


class _ConcreteFakeYamlState(FakeYamlState):
    """Concrete fake state suitable for direct ``TreeNode`` tests."""

    def pprint(self) -> str:
        """Return a tiny printable representation."""
        return str(self.node_id)


class _BranchKeys:
    def __init__(self, keys: list[int]) -> None:
        self._keys = keys

    def get_all(self) -> list[int]:
        """Return legal keys in configured order."""
        return self._keys


class _Dynamics:
    def __init__(self, keys: list[int]) -> None:
        self._keys = keys

    def legal_actions(self, state: object) -> _BranchKeys:
        """Return legal keys for ``state``."""
        del state
        return _BranchKeys(self._keys)


def _make_state(node_id: int) -> _ConcreteFakeYamlState:
    return _ConcreteFakeYamlState(
        node_id=node_id,
        children_by_id={node_id: []},
        turn=Color.WHITE,
    )


def _make_node(node_id: int = 1) -> TreeNode[Any, _ConcreteFakeYamlState]:
    return TreeNode[Any, _ConcreteFakeYamlState](
        state_handle_=MaterializedStateHandle(state_=_make_state(node_id)),
        tree_depth_=0,
        id_=node_id,
        parent_nodes_={},
    )


def test_fresh_tree_node_does_not_allocate_lazy_containers() -> None:
    """Fresh nodes should store no empty branch containers."""
    node = _make_node()

    assert node.branches_children_ is None
    assert node.non_opened_branches_ is None
    assert not node.has_child_links()
    assert node.child_link_count() == 0
    assert list(node.iter_child_links()) == []
    assert list(node.iter_child_nodes()) == []
    assert not node.has_unopened_branches()
    assert node.unopened_branch_count() == 0
    assert list(node.iter_unopened_branches()) == []


def test_read_only_helpers_do_not_allocate() -> None:
    """Read helpers should treat ``None`` storage as empty without materializing."""
    node = _make_node()

    assert node.child_for_branch(1) is None
    assert not node.has_child_for_branch(1)
    assert not node.contains_unopened_branch(1)
    assert list(node.iter_child_links()) == []
    assert list(node.iter_child_nodes()) == []
    assert list(node.iter_unopened_branches()) == []

    assert node.branches_children_ is None
    assert node.non_opened_branches_ is None


def test_child_mutation_allocates_only_when_needed() -> None:
    """Child mutation should allocate, then release storage when emptied."""
    node = _make_node()
    child = SimpleNamespace(id=2)

    node.set_child_for_branch(3, child)

    assert node.branches_children_ == {3: child}
    assert node.child_for_branch(3) is child
    assert list(node.iter_child_nodes()) == [child]

    node.remove_child_link(3)
    assert node.branches_children_ is None

    node.set_child_for_branch(5, None)
    assert node.branches_children_ == {5: None}
    node.clear_child_links()
    assert node.branches_children_ is None


def test_unopened_mutation_allocates_only_for_non_empty_values() -> None:
    """Unopened-branch mutation should normalize empty storage back to ``None``."""
    node = _make_node()

    node.set_unopened_branches([])
    assert node.non_opened_branches_ is None

    node.set_unopened_branches([1, 2])
    assert node.non_opened_branches_ == {1, 2}

    node.discard_unopened_branch(1)
    assert node.non_opened_branches_ == {2}

    node.discard_unopened_branch(2)
    assert node.non_opened_branches_ is None

    node.set_unopened_branches([3])
    node.clear_unopened_branches()
    assert node.non_opened_branches_ is None


def test_materializing_compatibility_properties_still_work() -> None:
    """Compatibility properties should materialize and support old-style mutation."""
    node = _make_node()
    child = SimpleNamespace(id=2)

    node.branches_children[1] = child
    node.non_opened_branches.add(2)

    assert node.branches_children_ == {1: child}
    assert node.non_opened_branches_ == {2}

    node.branches_children = {}
    node.non_opened_branches = set()

    assert node.branches_children_ is None
    assert node.non_opened_branches_ is None


def test_sync_opening_status_does_not_store_empty_unopened_set() -> None:
    """A fully opened node should keep unopened storage lazy after sync."""
    node = _make_node()
    node.set_child_for_branch(0, SimpleNamespace(id=2))

    sync_opening_status(node=node, dynamics=_Dynamics([0]))

    assert node.all_branches_generated
    assert node.non_opened_branches_ is None


def test_sync_opening_status_stores_only_openable_branches() -> None:
    """A partially opened node should store only currently openable legal branches."""
    node = _make_node()
    node.set_child_for_branch(0, SimpleNamespace(id=2))

    sync_opening_status(node=node, dynamics=_Dynamics([0, 1, 2]))

    assert not node.all_branches_generated
    assert node.non_opened_branches_ == {1, 2}
