"""Tests for registry-backed priority-check factory wiring."""

from collections.abc import Mapping
from dataclasses import dataclass
from random import Random
from typing import Any, Never

import pytest

from anemone.hooks import PriorityCheckFactory, SearchHooks
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningInstructor, OpeningType
from anemone.node_selector.priority_check.factory import create_priority_check
from anemone.node_selector.priority_check.registered_args import (
    RegisteredPriorityCheckArgs,
)


@dataclass
class DummyPriorityCheck:
    """Minimal priority-check object for identity assertions in tests."""


import valanga

from anemone.dynamics import SearchDynamics


class DummyDynamics(SearchDynamics[Any]):
    def legal_actions(self, state: Any) -> Never:
        raise RuntimeError("Not used in this test")

    def step(self, state: Any, action: valanga.BranchKey, *, depth: int) -> Never:
        raise RuntimeError("Not used in this test")

    def action_name(self, state: Any, action: valanga.BranchKey) -> str:
        return str(action)

    def action_from_name(self, state: Any, name: str):
        return name


def test_registered_priority_requires_hooks() -> None:
    """Registered checks should fail when no hooks are provided."""
    args = RegisteredPriorityCheckArgs(
        type=NodeSelectorType.PRIORITY_REGISTERED,
        name="x",
    )

    with pytest.raises(ValueError, match="requires hooks"):
        create_priority_check(
            args=args,
            random_generator=Random(0),
            hooks=None,
            opening_instructor=OpeningInstructor(
                OpeningType.ALL_CHILDREN,
                Random(0),
                dynamics=DummyDynamics(),
            ),
        )


def test_registered_priority_unknown_name() -> None:
    """Unknown registry names should raise a key error."""
    args = RegisteredPriorityCheckArgs(
        type=NodeSelectorType.PRIORITY_REGISTERED,
        name="missing",
    )

    with pytest.raises(KeyError, match="not found"):
        create_priority_check(
            args=args,
            random_generator=Random(0),
            hooks=SearchHooks(priority_check_registry={}),
            opening_instructor=OpeningInstructor(
                OpeningType.ALL_CHILDREN,
                Random(0),
                dynamics=DummyDynamics(),
            ),
        )


def test_registered_priority_factory_called() -> None:
    """A registered factory should be invoked and its value returned."""
    expected = DummyPriorityCheck()
    seen: dict[str, Any] = {}

    def factory(
        params: Mapping[str, Any],
        random_generator: Random,
        hooks: SearchHooks | None,
        opening_instructor: OpeningInstructor,
    ) -> DummyPriorityCheck:
        seen["params"] = params
        seen["random_generator"] = random_generator
        seen["hooks"] = hooks
        seen["opening_instructor"] = opening_instructor
        return expected

    args = RegisteredPriorityCheckArgs(
        type=NodeSelectorType.PRIORITY_REGISTERED,
        name="x",
        params={"alpha": 1},
    )
    random_generator = Random(42)
    opening_instructor = OpeningInstructor(
        OpeningType.ALL_CHILDREN, Random(3), dynamics=DummyDynamics()
    )
    registry: dict[str, PriorityCheckFactory] = {"x": factory}
    hooks = SearchHooks(priority_check_registry=registry)

    got = create_priority_check(
        args=args,
        random_generator=random_generator,
        hooks=hooks,
        opening_instructor=opening_instructor,
    )

    assert got is expected
    assert seen["params"] == {"alpha": 1}
    assert seen["random_generator"] is random_generator
    assert seen["hooks"] is hooks
    assert seen["opening_instructor"] is opening_instructor
