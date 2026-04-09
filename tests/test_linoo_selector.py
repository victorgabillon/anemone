"""Tests for the Linoo single-player node selector."""

from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace

import pytest
from valanga.evaluations import Certainty, Value

from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.factory import create, create_composed_node_selector
from anemone.node_selector.linoo import (
    Linoo,
    LinooArgs,
    LinooDirectValueUnavailableError,
    LinooIncompatibleObjectiveError,
)
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
    OpeningType,
)
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.objectives import AdversarialZeroSumObjective, SingleAgentMaxObjective


@dataclass
class _FakeEval:
    direct_value: Value | None
    exact: bool = False
    required_objective: object | None = None

    def has_exact_value(self) -> bool:
        return self.exact


@dataclass
class _MissingDirectValueEval:
    exact: bool = False
    required_objective: object | None = None

    def has_exact_value(self) -> bool:
        return self.exact


@dataclass
class _FakeNode:
    id: int
    tree_depth: int
    tree_evaluation: object
    all_branches_generated: bool = False
    state: object = field(default_factory=lambda: SimpleNamespace(turn="solo"))


class _FakeOpeningInstructor:
    def all_branches_to_open(self, node_to_open: _FakeNode) -> list[int]:
        _ = node_to_open
        return [0]


def _value(score: float) -> Value:
    return Value(score=score, certainty=Certainty.ESTIMATE, over_event=None)


def _node(
    node_id: int,
    depth: int,
    *,
    score: float | None = 0.0,
    opened: bool = False,
    exact: bool = False,
    eval_type: type[_FakeEval | _MissingDirectValueEval] = _FakeEval,
) -> _FakeNode:
    if eval_type is _FakeEval:
        evaluation: object = _FakeEval(
            direct_value=None if score is None else _value(score),
            exact=exact,
        )
    else:
        evaluation = _MissingDirectValueEval(exact=exact)
    return _FakeNode(
        id=node_id,
        tree_depth=depth,
        tree_evaluation=evaluation,
        all_branches_generated=opened,
    )


def _tree(
    *nodes: _FakeNode,
    objective: object | None = None,
) -> SimpleNamespace:
    root = min(nodes, key=lambda node: (node.tree_depth, node.id))
    if objective is None:
        objective = SingleAgentMaxObjective()

    root.tree_evaluation.required_objective = objective

    descendants: dict[int, dict[int, _FakeNode]] = {}
    for node in nodes:
        descendants.setdefault(node.tree_depth, {})[node.id] = node

    return SimpleNamespace(
        root_node=root,
        descendants=descendants,
        node_depth=lambda node: node.tree_depth - root.tree_depth,
    )


def _selected_node_id(instructions: OpeningInstructions[_FakeNode]) -> int:
    return next(iter(instructions.values())).node_to_open.id


def test_linoo_chooses_depth_with_minimum_opened_count_index() -> None:
    """Linoo should prefer the depth with the smallest opened-count index."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, opened=True),
        _node(11, 1, opened=True),
        _node(12, 1, score=100.0),
        _node(20, 2, opened=True),
        _node(21, 2, score=1.0),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    assert _selected_node_id(instructions) == 21


def test_linoo_chooses_best_direct_value_within_selected_depth() -> None:
    """Linoo should maximize direct value inside the selected depth."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=1.0),
        _node(11, 1, score=5.0),
        _node(12, 1, score=3.0),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    assert _selected_node_id(instructions) == 11


def test_linoo_breaks_equal_depth_indices_by_smaller_depth() -> None:
    """Equal depth indices should deterministically prefer the shallower depth."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, score=2.0),
        _node(10, 1, score=9.0),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    assert _selected_node_id(instructions) == 0


def test_linoo_breaks_equal_direct_values_by_smallest_node_id() -> None:
    """Equal direct values should deterministically prefer the smallest node id."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, opened=True),
        _node(7, 1, score=4.0),
        _node(3, 1, score=4.0),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    assert _selected_node_id(instructions) == 3


def test_linoo_raises_when_direct_value_is_unavailable() -> None:
    """Linoo should fail fast when a frontier node lacks direct_value access."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, eval_type=_MissingDirectValueEval),
    )

    with pytest.raises(
        LinooDirectValueUnavailableError,
        match="direct single-player value",
    ):
        selector.choose_node_and_branch_to_open(
            tree=tree,
            latest_tree_expansions=SimpleNamespace(),
        )


def test_linoo_raises_in_non_single_player_context() -> None:
    """Linoo should reject objectives outside the single-player max family."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=1.0),
        objective=AdversarialZeroSumObjective(),
    )

    with pytest.raises(
        LinooIncompatibleObjectiveError,
        match="SingleAgentMaxObjective",
    ):
        selector.choose_node_and_branch_to_open(
            tree=tree,
            latest_tree_expansions=SimpleNamespace(),
        )


def test_linoo_selector_factory_supports_direct_and_composed_creation() -> None:
    """Factory wiring should build Linoo both directly and inside composition."""
    opening_instructor = OpeningInstructor(
        opening_type=OpeningType.ALL_CHILDREN,
        random_generator=Random(0),
        dynamics=SimpleNamespace(),
    )

    selector = create(
        args=LinooArgs(type=NodeSelectorType.LINOO),
        opening_instructor=opening_instructor,
        random_generator=Random(0),
    )
    composed_selector = create_composed_node_selector(
        args=ComposedNodeSelectorArgs(
            type=NodeSelectorType.COMPOSED,
            priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
            base=LinooArgs(type=NodeSelectorType.LINOO),
        ),
        opening_instructor=opening_instructor,
        random_generator=Random(0),
        hooks=None,
    )

    assert isinstance(selector, Linoo)
    assert isinstance(composed_selector.base, Linoo)
