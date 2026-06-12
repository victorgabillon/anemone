"""Tests for structural opening-status helpers."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from anemone.nodes.opening_status import (
    has_openable_branches,
    is_fully_open_wrt_legal_actions,
    openable_branch_keys,
    opened_branch_keys,
)


@dataclass
class _BranchKeys:
    keys: list[int]

    def get_all(self) -> list[int]:
        """Return legal keys in configured order."""
        return self.keys


@dataclass
class _Dynamics:
    keys: list[int]

    def legal_actions(self, state: object) -> _BranchKeys:
        """Return legal keys for ``state``."""
        del state
        return _BranchKeys(self.keys)


@dataclass
class _Node:
    state: object = field(default_factory=object)
    branches_children: dict[int, Any | None] = field(default_factory=dict)


def test_unopened_node_reports_all_legal_branches_openable() -> None:
    """A node with no child links is openable on every legal branch."""
    node = _Node()
    dynamics = _Dynamics([0, 1, 2])

    assert opened_branch_keys(node) == set()
    assert openable_branch_keys(node=node, dynamics=dynamics) == (0, 1, 2)
    assert has_openable_branches(node=node, dynamics=dynamics)
    assert not is_fully_open_wrt_legal_actions(node=node, dynamics=dynamics)


def test_partially_opened_node_reports_remaining_legal_branches() -> None:
    """Only concrete child links count as already opened."""
    node = _Node(branches_children={1: SimpleNamespace(id=10)})
    dynamics = _Dynamics([0, 1, 2])

    assert opened_branch_keys(node) == {1}
    assert openable_branch_keys(node=node, dynamics=dynamics) == (0, 2)
    assert has_openable_branches(node=node, dynamics=dynamics)
    assert not is_fully_open_wrt_legal_actions(node=node, dynamics=dynamics)


def test_fully_opened_node_has_no_openable_branches() -> None:
    """A node is fully open when every legal branch has a child link."""
    node = _Node(
        branches_children={
            0: SimpleNamespace(id=10),
            1: SimpleNamespace(id=11),
            2: SimpleNamespace(id=12),
        }
    )
    dynamics = _Dynamics([0, 1, 2])

    assert openable_branch_keys(node=node, dynamics=dynamics) == ()
    assert not has_openable_branches(node=node, dynamics=dynamics)
    assert is_fully_open_wrt_legal_actions(node=node, dynamics=dynamics)


def test_openable_branches_preserve_legal_action_order() -> None:
    """Remaining branches follow dynamics order rather than sorted order."""
    node = _Node(branches_children={0: SimpleNamespace(id=10)})
    dynamics = _Dynamics([2, 0, 1])

    assert openable_branch_keys(node=node, dynamics=dynamics) == (2, 1)


def test_none_child_links_are_treated_as_unopened() -> None:
    """A branch mapped to ``None`` is still openable."""
    node = _Node(branches_children={0: None, 1: SimpleNamespace(id=11)})
    dynamics = _Dynamics([0, 1, 2])

    assert opened_branch_keys(node) == {1}
    assert openable_branch_keys(node=node, dynamics=dynamics) == (0, 2)
