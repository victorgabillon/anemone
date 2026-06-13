"""Tests for opening-instruction branch selection."""

from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace
from typing import Any

from anemone.node_selector.opening_instructions import OpeningInstructor, OpeningType


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
    id: int = 0
    state: object = field(default_factory=object)
    branches_children: dict[int, Any | None] = field(default_factory=dict)
    all_branches_generated: bool = False
    non_opened_branches: set[int] = field(default_factory=set)


def test_all_children_requests_only_unopened_legal_branches() -> None:
    """ALL_CHILDREN means all remaining legal branches."""
    node = _Node(branches_children={1: SimpleNamespace(id=10)})
    instructor = OpeningInstructor(
        opening_type=OpeningType.ALL_CHILDREN,
        random_generator=Random(0),
        dynamics=_Dynamics([0, 1, 2]),
    )

    assert instructor.all_branches_to_open(node_to_open=node) == (0, 2)
    assert not node.all_branches_generated


def test_all_children_returns_empty_when_no_branch_remains() -> None:
    """Opening instructors compute branches without mutating node status."""
    node = _Node(
        branches_children={
            0: SimpleNamespace(id=10),
            1: SimpleNamespace(id=11),
            2: SimpleNamespace(id=12),
        },
        all_branches_generated=False,
    )
    instructor = OpeningInstructor(
        opening_type=OpeningType.ALL_CHILDREN,
        random_generator=Random(0),
        dynamics=_Dynamics([0, 1, 2]),
    )

    assert instructor.all_branches_to_open(node_to_open=node) == ()
    assert not node.all_branches_generated
