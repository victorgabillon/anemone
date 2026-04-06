"""Tests for representative root-to-node path helpers."""

from dataclasses import dataclass, field

from valanga import Color

from anemone.nodes.utils import (
    a_branch_key_sequence_from_root,
    a_branch_str_sequence_from_root,
)
from tests.fake_yaml_game import FakeYamlState


class _ConcreteFakeYamlState(FakeYamlState):
    """Concrete fake state suitable for path helper tests."""

    def pprint(self) -> str:
        """Return a small printable representation."""
        return str(self.node_id)


@dataclass(eq=False, slots=True)
class _FakeNode:
    """Minimal node implementing the structural surface used by utils."""

    id: int
    tree_depth: int
    state: _ConcreteFakeYamlState
    parent_nodes: dict["_FakeNode", set[object]] = field(default_factory=dict)


class _FakeDynamics:
    """Minimal dynamics adapter for branch-string rendering tests."""

    __anemone_search_dynamics__ = True

    def action_name(self, state: _ConcreteFakeYamlState, action: object) -> str:
        """Return a stable debug label for the representative action."""
        return f"{state.node_id}->{action}"


def _make_state(node_id: int) -> _ConcreteFakeYamlState:
    """Create a simple fake state with no children."""
    return _ConcreteFakeYamlState(
        node_id=node_id,
        children_by_id={node_id: []},
        turn=Color.WHITE,
    )


def test_representative_branch_sequences_match_tree_path() -> None:
    """A normal tree should keep the old unique-path behavior."""
    root = _FakeNode(id=1, tree_depth=0, state=_make_state(10))
    mid = _FakeNode(
        id=2,
        tree_depth=1,
        state=_make_state(20),
        parent_nodes={root: {4}},
    )
    leaf = _FakeNode(
        id=3,
        tree_depth=2,
        state=_make_state(30),
        parent_nodes={mid: {7}},
    )

    assert a_branch_key_sequence_from_root(leaf) == ["4", "7"]
    assert a_branch_str_sequence_from_root(leaf, dynamics=_FakeDynamics()) == [
        "10->4",
        "20->7",
    ]


def test_representative_branch_sequence_uses_deterministic_branch_choice() -> None:
    """Same-parent multiedges should pick the deterministic representative branch."""
    root = _FakeNode(id=1, tree_depth=0, state=_make_state(10))
    child = _FakeNode(
        id=2,
        tree_depth=1,
        state=_make_state(20),
        parent_nodes={root: {5, 3}},
    )

    assert a_branch_key_sequence_from_root(child) == ["3"]
    assert a_branch_str_sequence_from_root(child, dynamics=_FakeDynamics()) == [
        "10->3"
    ]


def test_representative_branch_sequence_uses_deterministic_parent_choice() -> None:
    """Multiple parents should pick the deterministic representative parent."""
    parent_a = _FakeNode(id=1, tree_depth=0, state=_make_state(10))
    parent_b = _FakeNode(id=2, tree_depth=0, state=_make_state(20))
    child = _FakeNode(
        id=3,
        tree_depth=1,
        state=_make_state(30),
        parent_nodes={
            parent_b: {1},
            parent_a: {9},
        },
    )

    assert a_branch_key_sequence_from_root(child) == ["9"]
    assert a_branch_str_sequence_from_root(child, dynamics=_FakeDynamics()) == [
        "10->9"
    ]
