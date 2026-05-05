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
    terminal: bool = False
    required_objective: object | None = None

    def has_exact_value(self) -> bool:
        return self.exact

    def is_terminal(self) -> bool:
        return self.terminal


@dataclass
class _MissingDirectValueEval:
    exact: bool = False
    terminal: bool = False
    required_objective: object | None = None

    def has_exact_value(self) -> bool:
        return self.exact

    def is_terminal(self) -> bool:
        return self.terminal


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


class _MarkedStateObjective(SingleAgentMaxObjective[object]):
    """Objective that fails when non-selected marker states are evaluated."""

    def evaluate_value(self, value: Value, state: object) -> float:
        if getattr(state, "raise_on_eval", False):
            raise AssertionError
        return super().evaluate_value(value, state)


def _value(score: float) -> Value:
    return Value(score=score, certainty=Certainty.ESTIMATE, over_event=None)


def _node(
    node_id: int,
    depth: int,
    *,
    score: float | None = 0.0,
    opened: bool = False,
    exact: bool = False,
    terminal: bool = False,
    eval_type: type[_FakeEval | _MissingDirectValueEval] = _FakeEval,
) -> _FakeNode:
    if eval_type is _FakeEval:
        evaluation: object = _FakeEval(
            direct_value=None if score is None else _value(score),
            exact=exact,
            terminal=terminal,
        )
    else:
        evaluation = _MissingDirectValueEval(exact=exact, terminal=terminal)
    return _FakeNode(
        id=node_id,
        tree_depth=depth,
        tree_evaluation=evaluation,
        all_branches_generated=opened,
        state=SimpleNamespace(
            turn="solo",
            is_game_over=lambda: terminal,
        ),
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


def test_linoo_selection_report_describes_candidate_depth_table() -> None:
    """The latest report should expose the full diagnostic depth table."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, opened=True),
        _node(11, 1, opened=True),
        _node(12, 1, score=100.0),
        _node(20, 2, opened=True),
        _node(21, 2, score=1.0),
        _node(22, 2, score=3.0),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    report = selector.latest_selection_report
    assert report is not None
    assert _selected_node_id(instructions) == 22
    assert report.selected_depth == 2
    assert report.selected_node_id == 22
    assert report.selected_node_direct_value == 3.0
    assert report.selected_depth_selection_index == 3
    assert report.depth_row_count == 3
    assert report.total_nodes_scanned == 7
    assert report.frontier_nodes_scanned == 3
    assert report.uncached_terminal_candidates == 0
    assert report.collect_frontier_state_s is not None
    assert report.collect_frontier_state_s >= 0.0
    assert report.choose_depth_s is not None
    assert report.choose_depth_s >= 0.0
    assert report.choose_node_s is not None
    assert report.choose_node_s >= 0.0
    assert report.make_report_s is not None
    assert report.make_report_s >= 0.0
    assert report.total_s is not None
    assert report.total_s >= 0.0
    assert tuple(row.depth for row in report.depth_rows) == (2, 1, 0)
    active_rows = tuple(row for row in report.depth_rows if row.active)
    assert tuple((row.selection_index, row.depth) for row in active_rows) == tuple(
        sorted((row.selection_index, row.depth) for row in active_rows)
    )

    rows_by_depth = {row.depth: row for row in report.depth_rows}
    assert set(rows_by_depth) == {0, 1, 2}
    assert rows_by_depth[0].total_nodes == 1
    assert rows_by_depth[0].opened_count == 1
    assert rows_by_depth[0].frontier_count == 0
    assert rows_by_depth[0].terminal_count == 0
    assert rows_by_depth[0].exact_count == 0
    assert rows_by_depth[0].uncached_terminal_candidates == 0
    assert rows_by_depth[0].non_openable_count == 0
    assert rows_by_depth[0].selection_index is None
    assert rows_by_depth[0].active is False
    assert rows_by_depth[0].selected is False
    assert rows_by_depth[1].total_nodes == 3
    assert rows_by_depth[1].opened_count == 2
    assert rows_by_depth[1].frontier_count == 1
    assert rows_by_depth[1].terminal_count == 0
    assert rows_by_depth[1].exact_count == 0
    assert rows_by_depth[1].uncached_terminal_candidates == 0
    assert rows_by_depth[1].non_openable_count == 0
    assert rows_by_depth[1].selection_index == 4
    assert rows_by_depth[1].active is True
    assert rows_by_depth[1].selected is False
    assert rows_by_depth[2].total_nodes == 3
    assert rows_by_depth[2].opened_count == 1
    assert rows_by_depth[2].frontier_count == 2
    assert rows_by_depth[2].terminal_count == 0
    assert rows_by_depth[2].exact_count == 0
    assert rows_by_depth[2].uncached_terminal_candidates == 0
    assert rows_by_depth[2].non_openable_count == 0
    assert rows_by_depth[2].selection_index == 3
    assert rows_by_depth[2].active is True
    assert rows_by_depth[2].selected is True
    assert report.depth_rows[0].depth == report.selected_depth
    assert report.format_depth_table().splitlines()[0].split() == [
        "depth",
        "total",
        "opened",
        "frontier",
        "terminal",
        "exact",
        "uncached_terminal",
        "non_openable",
        "index",
        "selected",
    ]


def test_linoo_selection_report_keeps_depths_without_frontier() -> None:
    """Inactive depths should stay visible without influencing selection."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, opened=True),
        _node(11, 1, opened=True),
        _node(20, 2, score=3.0),
        _node(30, 3, score=8.0, exact=True),
        _node(40, 4, score=1.0, exact=True, terminal=True),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    report = selector.latest_selection_report
    assert report is not None
    assert _selected_node_id(instructions) == 20
    rows_by_depth = {row.depth: row for row in report.depth_rows}
    assert set(rows_by_depth) == {0, 1, 2, 3, 4}
    assert rows_by_depth[1].total_nodes == 2
    assert rows_by_depth[1].opened_count == 2
    assert rows_by_depth[1].frontier_count == 0
    assert rows_by_depth[1].terminal_count == 0
    assert rows_by_depth[1].exact_count == 0
    assert rows_by_depth[1].uncached_terminal_candidates == 0
    assert rows_by_depth[1].non_openable_count == 0
    assert rows_by_depth[1].selection_index is None
    assert rows_by_depth[1].active is False
    assert rows_by_depth[1].selected is False
    assert rows_by_depth[3].total_nodes == 1
    assert rows_by_depth[3].opened_count == 0
    assert rows_by_depth[3].frontier_count == 0
    assert rows_by_depth[3].terminal_count == 0
    assert rows_by_depth[3].exact_count == 1
    assert rows_by_depth[3].uncached_terminal_candidates == 0
    assert rows_by_depth[3].non_openable_count == 0
    assert rows_by_depth[3].selection_index is None
    assert rows_by_depth[3].active is False
    assert rows_by_depth[3].selected is False
    assert rows_by_depth[4].total_nodes == 1
    assert rows_by_depth[4].opened_count == 0
    assert rows_by_depth[4].frontier_count == 0
    assert rows_by_depth[4].terminal_count == 1
    assert rows_by_depth[4].exact_count == 0
    assert rows_by_depth[4].uncached_terminal_candidates == 0
    assert rows_by_depth[4].non_openable_count == 0
    assert rows_by_depth[4].selection_index is None
    assert rows_by_depth[4].active is False
    assert rows_by_depth[4].selected is False
    assert rows_by_depth[2].active is True
    assert rows_by_depth[2].selected is True


def test_linoo_selection_report_classifies_terminal_nodes_before_opened_or_exact() -> (
    None
):
    """Terminal leaves should not be hidden as opened/exact/frontier nodes."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=1.0),
        _node(20, 2, score=2.0, terminal=True),
        _node(21, 2, score=3.0, terminal=True, opened=True, exact=True),
    )

    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    report = selector.latest_selection_report
    assert report is not None
    rows_by_depth = {row.depth: row for row in report.depth_rows}
    assert rows_by_depth[2].total_nodes == 2
    assert rows_by_depth[2].opened_count == 0
    assert rows_by_depth[2].frontier_count == 0
    assert rows_by_depth[2].terminal_count == 2
    assert rows_by_depth[2].exact_count == 0
    assert rows_by_depth[2].uncached_terminal_candidates == 0
    assert rows_by_depth[2].non_openable_count == 0
    assert rows_by_depth[2].selection_index is None
    assert rows_by_depth[2].selected is False


def test_linoo_does_not_call_state_is_game_over_for_terminality() -> None:
    """Linoo should use only cached tree-evaluation terminality."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    terminal_node = _node(20, 2, score=2.0, terminal=True)

    def _raise_if_called() -> bool:
        raise AssertionError

    terminal_node.state = SimpleNamespace(turn="solo", is_game_over=_raise_if_called)
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=1.0),
        terminal_node,
    )

    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    report = selector.latest_selection_report
    assert report is not None
    rows_by_depth = {row.depth: row for row in report.depth_rows}
    assert rows_by_depth[2].terminal_count == 1


def test_linoo_reports_uncached_terminal_candidates() -> None:
    """Openable-looking nodes without direct values should be counted."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=None),
        _node(11, 1, score=4.0),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    report = selector.latest_selection_report
    assert report is not None
    assert _selected_node_id(instructions) == 11
    assert report.uncached_terminal_candidates == 1
    rows_by_depth = {row.depth: row for row in report.depth_rows}
    assert rows_by_depth[1].frontier_count == 1
    assert rows_by_depth[1].uncached_terminal_candidates == 1
    assert rows_by_depth[1].non_openable_count == 0


def test_linoo_report_does_not_evaluate_non_selected_depths() -> None:
    """Report creation should not rescan frontier nodes to find debug bests."""
    selector = Linoo(opening_instructor=_FakeOpeningInstructor())
    non_selected_node = _node(20, 2, score=100.0)
    non_selected_node.state = SimpleNamespace(turn="solo", raise_on_eval=True)
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=1.0),
        non_selected_node,
        objective=_MarkedStateObjective(),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    report = selector.latest_selection_report
    assert report is not None
    assert _selected_node_id(instructions) == 10
    assert report.selected_node_id == 10
    rows_by_depth = {row.depth: row for row in report.depth_rows}
    assert rows_by_depth[2].active is True
    assert rows_by_depth[2].selected is False


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
