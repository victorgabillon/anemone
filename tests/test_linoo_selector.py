"""Tests for the Linoo single-player node selector."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace

import pytest
from valanga.evaluations import Certainty, Value

from anemone.checkpoints.payloads import (
    LinooDepthStatsCheckpointPayload,
    LinooNodeStateCheckpointPayload,
    LinooSelectorCheckpointPayload,
)
from anemone.node_evaluation.common.value_candidate import (
    ValueCandidate,
    ValueCandidateSource,
)
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.factory import create, create_composed_node_selector
from anemone.node_selector.linoo import (
    Linoo,
    LinooArgs,
    LinooDepthSelectionPolicy,
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
from anemone.tree_manager.tree_expander import TreeExpansion, TreeExpansions


@dataclass
class _FakeEval:
    direct_value: Value | None
    effective_value: Value | None = None
    effective_value_source: ValueCandidateSource = ValueCandidateSource.NONE
    exact: bool = False
    terminal: bool = False
    required_objective: object | None = None

    def has_exact_value(self) -> bool:
        return self.exact

    def is_terminal(self) -> bool:
        return self.terminal

    def get_effective_value_candidate(self) -> ValueCandidate:
        if self.effective_value is not None:
            return ValueCandidate(
                value=self.effective_value,
                source=self.effective_value_source,
            )
        if self.direct_value is not None:
            return ValueCandidate.direct(self.direct_value)
        return ValueCandidate.none()


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


@dataclass
class _ExplodingStateNode:
    id: int
    tree_depth: int
    tree_evaluation: object
    all_branches_generated: bool = False

    @property
    def state(self) -> object:
        raise AssertionError


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


class _StatefulOnlyObjective(SingleAgentMaxObjective[object]):
    """Objective that forces Linoo through the stateful fallback path."""

    evaluate_value_without_state = None

    def evaluate_value(self, value: Value, state: object) -> float:
        state.evaluated = True
        return value.score + state.priority_bonus


class _StubRandom(Random):
    """Deterministic RNG stub for Zipf interval tests."""

    def __init__(self, values: list[float] | None = None) -> None:
        super().__init__(0)
        self._values = [*([0.0] if values is None else values)]
        self._index = 0

    def random(self) -> float:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


def _value(score: float) -> Value:
    return Value(score=score, certainty=Certainty.ESTIMATE, over_event=None)


def _node(
    node_id: int,
    depth: int,
    *,
    score: float | None = 0.0,
    effective_score: float | None = None,
    effective_source: ValueCandidateSource = ValueCandidateSource.TREE_CHILD,
    opened: bool = False,
    exact: bool = False,
    terminal: bool = False,
    eval_type: type[_FakeEval | _MissingDirectValueEval] = _FakeEval,
) -> _FakeNode:
    if eval_type is _FakeEval:
        evaluation: object = _FakeEval(
            direct_value=None if score is None else _value(score),
            effective_value=None
            if effective_score is None
            else _value(effective_score),
            effective_value_source=effective_source
            if effective_score is not None
            else ValueCandidateSource.NONE,
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
        nodes_count=len(nodes),
        node_depth=lambda node: node.tree_depth - root.tree_depth,
    )


def _selected_node_id(instructions: OpeningInstructions[_FakeNode]) -> int:
    return next(iter(instructions.values())).node_to_open.id


def _selector(
    *,
    random_values: list[float] | None = None,
    depth_selection_policy: LinooDepthSelectionPolicy = "inverse_depth",
) -> Linoo[_FakeNode]:
    return Linoo(
        opening_instructor=_FakeOpeningInstructor(),
        random_generator=_StubRandom(random_values),
        depth_selection_policy=depth_selection_policy,
    )


def _scan_selected_depth(tree: SimpleNamespace) -> int:
    """Return the depth selected by the original scan-based Linoo policy."""
    depth_state_by_depth: dict[int, tuple[int, list[_FakeNode]]] = {}
    for absolute_tree_depth in tree.descendants:
        for node in tree.descendants[absolute_tree_depth].values():
            depth = tree.node_depth(node)
            opened_count, frontier_nodes = depth_state_by_depth.setdefault(
                depth,
                (0, []),
            )
            if node.tree_evaluation.is_terminal():
                continue
            if node.all_branches_generated:
                depth_state_by_depth[depth] = (opened_count + 1, frontier_nodes)
                continue
            if node.tree_evaluation.has_exact_value():
                continue
            if node.tree_evaluation.direct_value is None:
                continue
            frontier_nodes.append(node)

    frontier_by_depth = {
        depth: frontier_nodes
        for depth, (_opened_count, frontier_nodes) in depth_state_by_depth.items()
        if frontier_nodes
    }
    return min(
        frontier_by_depth,
        key=lambda depth: (depth_state_by_depth[depth][0] * (depth + 1), depth),
    )


def test_linoo_chooses_depth_with_minimum_opened_count_index() -> None:
    """Linoo should prefer the depth with the smallest opened-count index."""
    selector = _selector(depth_selection_policy="opened_count_depth_index")
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
    selector = _selector(depth_selection_policy="opened_count_depth_index")
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
    assert report.selected_node_candidate_value == 3.0
    assert report.selected_node_priority == 3.0
    assert report.selected_node_rank == 1
    assert report.ranked_candidate_count == 2
    assert report.node_selection_policy == "zipf_rank"
    assert report.depth_selection_policy == "opened_count_depth_index"
    assert report.selected_depth_selection_index == 3
    assert report.selected_depth_selection_weight is None
    assert report.selected_depth_selection_probability is None
    assert report.depth_row_count == 3
    assert report.total_nodes_scanned == 7
    assert report.state_rebuilt is True
    assert report.nodes_incrementally_updated == 0
    assert report.frontier_nodes_scanned == 3
    assert report.uncached_terminal_candidates == 0
    assert report.selected_depth_frontier_count == 2
    assert report.stale_candidates_skipped == 0
    assert report.heap_candidates_registered == 2
    assert report.heap_update_candidate_count == 2
    assert report.heap_update_push_count == 2
    assert report.heap_update_pop_count == 2
    assert report.heap_update_stale_skip_count == 0
    assert report.heap_update_signature_check_count == 2
    assert report.heap_update_signature_recompute_count == 2
    assert report.heap_update_version_mismatch_count == 0
    assert report.heap_update_priority_state_free_count == 2
    assert report.heap_update_priority_stateful_fallback_count == 0
    assert report.heap_update_candidate_direct_count == 2
    assert report.heap_update_candidate_tree_count == 0
    assert report.heap_update_candidate_unknown_count == 0
    assert report.heap_update_total_heap_entries == 2
    assert report.heap_update_max_heap_size == 2
    assert report.heap_update_depth_count == 0
    assert report.heap_update_frontier_node_count_seen == 2
    assert report.collect_frontier_state_s is not None
    assert report.collect_frontier_state_s >= 0.0
    assert report.choose_depth_s is not None
    assert report.choose_depth_s >= 0.0
    assert report.heap_update_s is not None
    assert report.heap_update_s >= 0.0
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
    assert rows_by_depth[1].selection_weight is None
    assert rows_by_depth[1].selection_probability is None
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
    assert rows_by_depth[2].selection_weight is None
    assert rows_by_depth[2].selection_probability is None
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
        "weight",
        "probability",
        "selected",
    ]


def test_linoo_defaults_to_inverse_depth_sampling() -> None:
    """The default depth policy should sample active depths by inverse depth."""
    selector = _selector(random_values=[0.7])
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

    report = selector.latest_selection_report
    assert report is not None
    assert _selected_node_id(instructions) == 21
    assert report.depth_selection_policy == "inverse_depth"
    assert report.selected_depth == 2
    assert report.selected_depth_selection_weight == pytest.approx(1.0 / 3.0)
    assert report.selected_depth_selection_probability == pytest.approx(0.4)
    rows_by_depth = {row.depth: row for row in report.depth_rows}
    assert rows_by_depth[1].selection_weight == pytest.approx(0.5)
    assert rows_by_depth[1].selection_probability == pytest.approx(0.6)
    assert rows_by_depth[2].selection_weight == pytest.approx(1.0 / 3.0)
    assert rows_by_depth[2].selection_probability == pytest.approx(0.4)


def test_linoo_inverse_depth_keeps_all_active_depths_possible() -> None:
    """Inverse-depth sampling should not exclude deeper frontier depths."""
    shallow_selector = _selector(random_values=[0.0])
    deep_selector = _selector(random_values=[0.999999])
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=1.0),
        _node(20, 2, score=2.0),
    )

    shallow_instructions = shallow_selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )
    deep_instructions = deep_selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    assert _selected_node_id(shallow_instructions) == 10
    assert _selected_node_id(deep_instructions) == 20


def test_linoo_rejects_unknown_depth_selection_policy() -> None:
    """Invalid depth policies should fail at construction with a clear error."""
    with pytest.raises(ValueError, match="Invalid Linoo depth-selection policy"):
        Linoo(
            opening_instructor=_FakeOpeningInstructor(),
            random_generator=_StubRandom(),
            depth_selection_policy="bogus",  # type: ignore[arg-type]
        )


def test_linoo_enters_optional_diagnostic_phase_context() -> None:
    """Linoo should expose nested diagnostic phase scopes without hard dependencies."""
    entered_phases: list[str] = []

    @contextmanager
    def phase_context(phase: str) -> Iterator[None]:
        entered_phases.append(phase)
        yield

    selector = _selector()
    selector._diagnostic_phase_context = phase_context
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=1.0),
        _node(11, 1, score=2.0),
    )

    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    assert "select.collect" in entered_phases
    assert "select.choose_depth" in entered_phases
    assert "select.heap_update" in entered_phases
    assert "select.heap_update.signature" in entered_phases
    assert "select.heap_update.push" in entered_phases
    assert "select.choose_node" in entered_phases
    assert "select.heap_update.pop" in entered_phases
    assert "select.heap_update.stale_check" in entered_phases
    assert "select.report" in entered_phases


def test_linoo_updates_latest_expansion_incrementally() -> None:
    """After the initial rebuild, Linoo should refresh only touched nodes."""
    selector = _selector(depth_selection_policy="opened_count_depth_index")
    root = _node(0, 0, opened=True)
    first = _node(10, 1, score=5.0)
    second = _node(11, 1, score=4.0)
    tree = _tree(root, first, second)

    first_instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    first_report = selector.latest_selection_report
    assert first_report is not None
    assert _selected_node_id(first_instructions) == 10
    assert first_report.state_rebuilt is True
    assert first_report.total_nodes_scanned == 3

    first.all_branches_generated = True
    child = _node(20, 2, score=100.0)
    tree.descendants.setdefault(child.tree_depth, {})[child.id] = child
    tree.nodes_count += 1
    expansions: TreeExpansions[_FakeNode] = TreeExpansions()
    expansions.record_creation(
        TreeExpansion(
            child_node=child,
            parent_node=first,
            state_modifications=None,
            creation_child_node=True,
            branch_key=0,
        )
    )

    second_instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=expansions,
    )

    second_report = selector.latest_selection_report
    assert second_report is not None
    assert _selected_node_id(second_instructions) == 20
    assert second_report.state_rebuilt is False
    assert second_report.nodes_incrementally_updated == 2
    assert second_report.total_nodes_scanned == 2
    rows_by_depth = {row.depth: row for row in second_report.depth_rows}
    assert rows_by_depth[1].opened_count == 1
    assert rows_by_depth[1].frontier_count == 1
    assert rows_by_depth[2].frontier_count == 1


def test_linoo_sparse_state_omits_default_opened_nodes() -> None:
    """Default opened nodes should be counted without per-node state objects."""
    selector = _selector()
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, opened=True),
        _node(11, 1, score=4.0),
    )

    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    assert selector._node_state_or_none(0) is None
    assert selector._node_state_or_none(10) is None
    frontier_state = selector._node_state_or_none(11)
    assert frontier_state is not None
    assert frontier_state.node_id == 11
    assert not hasattr(frontier_state, "node")
    assert frontier_state.status == "frontier"
    assert set(selector._node_state_by_id) == {11}


def test_linoo_tracks_last_selected_node_id_without_node_reference() -> None:
    """The latest selected node cache should retain only the public node id."""
    selector = _selector()
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=4.0),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    assert _selected_node_id(instructions) == 10
    assert selector._last_selected_node_id == 10
    assert not hasattr(selector, "_last_selected_node")


def test_linoo_candidate_heap_stores_node_ids_only() -> None:
    """Candidate heaps should not retain live node objects after registration."""
    selector = _selector()
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=5.0),
        _node(11, 1, score=4.0),
    )

    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    heap_entries = selector._candidates_by_depth[1]
    assert heap_entries
    assert all(len(entry) == 3 for entry in heap_entries)
    assert all(
        not any(isinstance(value, _FakeNode) for value in entry)
        for entry in heap_entries
    )


def test_linoo_builds_nodes_by_id_once_per_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One selection should build a node-id lookup once, not scan per candidate."""
    selector = _selector()
    tree = _tree(
        _node(0, 0, opened=True),
        *[_node(100 + index, 1, score=float(index)) for index in range(1, 40)],
    )
    call_count = 0
    original = selector._nodes_by_id_from_tree

    def _counted_nodes_by_id(tree_arg: SimpleNamespace) -> dict[int, _FakeNode]:
        nonlocal call_count
        call_count += 1
        return original(tree_arg)

    monkeypatch.setattr(selector, "_nodes_by_id_from_tree", _counted_nodes_by_id)

    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    assert call_count == 1


def test_linoo_sparse_state_discards_node_that_returns_to_default() -> None:
    """Reclassification should drop materialized state once it becomes opened."""
    selector = _selector(depth_selection_policy="opened_count_depth_index")
    root = _node(0, 0, opened=True)
    first = _node(10, 1, score=5.0)
    tree = _tree(root, first)

    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )
    assert selector._node_state_or_none(10) is not None

    first.all_branches_generated = True
    expansions: TreeExpansions[_FakeNode] = TreeExpansions()
    expansions.record_connection(
        TreeExpansion(
            child_node=first,
            parent_node=root,
            state_modifications=None,
            creation_child_node=False,
            branch_key=0,
        )
    )
    child = _node(20, 2, score=100.0)
    tree.descendants.setdefault(child.tree_depth, {})[child.id] = child
    tree.nodes_count += 1
    expansions.record_creation(
        TreeExpansion(
            child_node=child,
            parent_node=first,
            state_modifications=None,
            creation_child_node=True,
            branch_key=0,
        )
    )

    selector.choose_node_and_branch_to_open(
        tree=tree, latest_tree_expansions=expansions
    )

    assert selector._node_state_or_none(10) is None
    assert selector._node_state_or_none(20) is not None


def test_linoo_selection_report_keeps_depths_without_frontier() -> None:
    """Inactive depths should stay visible without influencing selection."""
    selector = _selector()
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
    selector = _selector()
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
    selector = _selector()
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


def test_linoo_heap_push_does_not_access_state_for_single_agent_max() -> None:
    """State-free objectives should keep heap priority projection state-free."""
    selector = _selector()
    root = _node(0, 0, opened=True)
    frontier = _ExplodingStateNode(
        id=10,
        tree_depth=1,
        tree_evaluation=_FakeEval(direct_value=_value(1.0)),
    )
    tree = _tree(root, frontier)

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    assert _selected_node_id(instructions) == 10
    report = selector.latest_selection_report
    assert report is not None
    assert report.heap_update_priority_state_free_count == 1
    assert report.heap_update_priority_stateful_fallback_count == 0


def test_linoo_reports_uncached_terminal_candidates() -> None:
    """Openable-looking nodes without direct values should be counted."""
    selector = _selector()
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


def test_linoo_does_not_evaluate_non_selected_depths() -> None:
    """Only selected-depth frontier candidates should enter the objective heap."""
    selector = _selector(depth_selection_policy="opened_count_depth_index")
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
    assert report.heap_candidates_registered == 1
    rows_by_depth = {row.depth: row for row in report.depth_rows}
    assert rows_by_depth[2].active is True
    assert rows_by_depth[2].selected is False


def test_linoo_ranks_candidates_by_effective_value_over_direct_value() -> None:
    """Partially opened frontier nodes should rank by effective candidate value."""
    selector = _selector()
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=0.1, effective_score=0.9),
        _node(11, 1, score=0.8, effective_score=0.2),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    assert _selected_node_id(instructions) == 10
    report = selector.latest_selection_report
    assert report is not None
    assert report.selected_node_direct_value == 0.1
    assert report.selected_node_candidate_value == 0.9
    assert report.selected_node_priority == 0.9
    assert report.heap_update_candidate_direct_count == 0
    assert report.heap_update_candidate_tree_count == 2


def test_linoo_depth_selection_matches_scan_based_policy() -> None:
    """Zipf node choice should not change the selected frontier depth."""
    selector = _selector(
        random_values=[0.999999],
        depth_selection_policy="opened_count_depth_index",
    )
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, opened=True),
        _node(11, 1, score=5.0),
        _node(12, 1, score=7.0),
        _node(20, 2, opened=True),
        _node(21, 2, score=100.0),
        _node(22, 2, score=100.0),
        _node(30, 3, score=200.0, exact=True),
        _node(40, 4, score=300.0, terminal=True),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    report = selector.latest_selection_report
    assert report is not None
    assert report.selected_depth == _scan_selected_depth(tree)
    assert _selected_node_id(instructions) in {11, 12}


def test_linoo_heap_updates_when_touched_frontier_direct_value_changes() -> None:
    """Touched direct-value changes should replace older heap priorities lazily."""
    selector = _selector()
    first = _node(10, 1, score=3.0)
    second = _node(11, 1, score=2.0)
    tree = _tree(_node(0, 0, opened=True), first, second)

    assert (
        _selected_node_id(
            selector.choose_node_and_branch_to_open(
                tree=tree,
                latest_tree_expansions=SimpleNamespace(),
            )
        )
        == 10
    )

    first.tree_evaluation.direct_value = _value(1.0)
    expansions: TreeExpansions[_FakeNode] = TreeExpansions()
    expansions.record_connection(
        TreeExpansion(
            child_node=first,
            parent_node=None,
            state_modifications=None,
            creation_child_node=False,
            branch_key=0,
        )
    )
    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=expansions,
    )

    assert _selected_node_id(instructions) == 11
    report = selector.latest_selection_report
    assert report is not None
    assert report.state_rebuilt is False
    assert report.nodes_incrementally_updated == 1
    assert report.stale_candidates_skipped == 1
    assert report.heap_candidates_registered == 1
    assert report.heap_update_candidate_count == 2
    assert report.heap_update_push_count == 1
    assert report.heap_update_stale_skip_count == 1
    assert report.heap_update_version_mismatch_count == 1


def test_linoo_heap_updates_when_touched_frontier_effective_value_changes() -> None:
    """Touched effective-value changes should replace heap priorities lazily."""
    selector = _selector()
    first = _node(10, 1, score=3.0, effective_score=3.0)
    second = _node(11, 1, score=2.0, effective_score=2.0)
    tree = _tree(_node(0, 0, opened=True), first, second)

    assert (
        _selected_node_id(
            selector.choose_node_and_branch_to_open(
                tree=tree,
                latest_tree_expansions=TreeExpansions(),
            )
        )
        == 10
    )

    first.tree_evaluation.effective_value = _value(1.0)
    expansions: TreeExpansions[_FakeNode] = TreeExpansions()
    expansions.record_connection(
        TreeExpansion(
            child_node=first,
            parent_node=None,
            state_modifications=None,
            creation_child_node=False,
            branch_key=0,
        )
    )
    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=expansions,
    )

    assert _selected_node_id(instructions) == 11
    report = selector.latest_selection_report
    assert report is not None
    assert report.state_rebuilt is False
    assert report.nodes_incrementally_updated == 1
    assert report.stale_candidates_skipped == 1
    assert report.heap_candidates_registered == 1
    assert report.heap_update_push_count == 1
    assert report.heap_update_stale_skip_count == 1
    assert report.heap_update_version_mismatch_count == 1


def test_linoo_priority_falls_back_to_stateful_objective() -> None:
    """Objectives without a state-free projector should still receive node state."""
    selector = _selector()
    lower_state = SimpleNamespace(turn="solo", priority_bonus=0.0, evaluated=False)
    higher_state = SimpleNamespace(turn="solo", priority_bonus=2.0, evaluated=False)
    lower = _node(10, 1, score=1.0)
    higher = _node(11, 1, score=0.5)
    lower.state = lower_state
    higher.state = higher_state
    tree = _tree(
        _node(0, 0, opened=True),
        lower,
        higher,
        objective=_StatefulOnlyObjective(),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    assert _selected_node_id(instructions) == 11
    assert lower_state.evaluated is True
    assert higher_state.evaluated is True
    report = selector.latest_selection_report
    assert report is not None
    assert report.heap_update_priority_state_free_count == 0
    assert report.heap_update_priority_stateful_fallback_count == 2


def test_linoo_checkpoint_payload_roundtrips_initialized_cache() -> None:
    """A restored Linoo cache should select without a full rebuild."""
    objective = SingleAgentMaxObjective()
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=5.0),
        _node(11, 1, score=4.0),
        objective=objective,
    )
    selector = _selector()

    first_instructions = selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )
    payload = selector.build_checkpoint_payload(objective)
    assert payload is not None
    assert payload.last_selected_node_id == _selected_node_id(first_instructions)

    restored_selector = _selector()
    restored = restored_selector.restore_from_checkpoint_payload(
        tree=tree,
        objective=objective,
        payload=payload,
    )
    restored_instructions = restored_selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    assert restored is True
    assert _selected_node_id(restored_instructions) == _selected_node_id(
        first_instructions
    )
    assert restored_selector.latest_selection_report is not None
    assert restored_selector.latest_selection_report.state_rebuilt is False
    assert restored_selector._node_state_or_none(0) is None
    assert restored_selector._node_state_or_none(10) is not None
    assert restored_selector._last_selected_node_id == _selected_node_id(
        restored_instructions
    )


def test_linoo_checkpoint_restore_accepts_missing_default_node_state() -> None:
    """Sparse selector payloads may omit default opened nodes."""
    objective = SingleAgentMaxObjective()
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=5.0),
        objective=objective,
    )
    selector = _selector()
    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )
    payload = selector.build_checkpoint_payload(objective)
    assert payload is not None

    payload.node_states = [
        node_state for node_state in payload.node_states if node_state.node_id != 0
    ]

    restored_selector = _selector()
    restored = restored_selector.restore_from_checkpoint_payload(
        tree=tree,
        objective=objective,
        payload=payload,
    )

    assert restored is True
    assert restored_selector._node_state_or_none(0) is None
    assert restored_selector._node_state_or_none(10) is not None


def test_linoo_checkpoint_restore_rejects_missing_non_default_node_state() -> None:
    """Missing non-default node states should still invalidate stale payloads."""
    objective = SingleAgentMaxObjective()
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=5.0),
        objective=objective,
    )
    payload = LinooSelectorCheckpointPayload(
        depth_stats=[
            LinooDepthStatsCheckpointPayload(
                depth=0,
                total_nodes=1,
                opened_count=1,
                frontier_count=0,
                terminal_count=0,
                exact_count=0,
                uncached_terminal_candidates=0,
                non_openable_count=0,
            ),
            LinooDepthStatsCheckpointPayload(
                depth=1,
                total_nodes=1,
                opened_count=0,
                frontier_count=1,
                terminal_count=0,
                exact_count=0,
                uncached_terminal_candidates=0,
                non_openable_count=0,
            ),
        ],
        node_states=[],
    )

    restored_selector = _selector()
    restored = restored_selector.restore_from_checkpoint_payload(
        tree=tree,
        objective=objective,
        payload=payload,
    )
    instructions = restored_selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    assert restored is False
    assert _selected_node_id(instructions) == 10
    assert restored_selector.latest_selection_report is not None
    assert restored_selector.latest_selection_report.state_rebuilt is True


def test_linoo_checkpoint_restore_skips_explicit_default_node_state() -> None:
    """Old dense checkpoints with explicit opened entries should load sparsely."""
    objective = SingleAgentMaxObjective()
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=5.0),
        objective=objective,
    )
    payload = LinooSelectorCheckpointPayload(
        depth_stats=[
            LinooDepthStatsCheckpointPayload(
                depth=0,
                total_nodes=1,
                opened_count=1,
                frontier_count=0,
                terminal_count=0,
                exact_count=0,
                uncached_terminal_candidates=0,
                non_openable_count=0,
            ),
            LinooDepthStatsCheckpointPayload(
                depth=1,
                total_nodes=1,
                opened_count=0,
                frontier_count=1,
                terminal_count=0,
                exact_count=0,
                uncached_terminal_candidates=0,
                non_openable_count=0,
            ),
        ],
        node_states=[
            LinooNodeStateCheckpointPayload(node_id=0, depth=0, status="opened"),
            LinooNodeStateCheckpointPayload(node_id=10, depth=1, status="frontier"),
        ],
    )

    restored_selector = _selector()
    restored = restored_selector.restore_from_checkpoint_payload(
        tree=tree,
        objective=objective,
        payload=payload,
    )

    assert restored is True
    assert restored_selector._node_state_or_none(0) is None
    assert restored_selector._node_state_or_none(10) is not None


def test_linoo_checkpoint_restore_accepts_legacy_node_reference_payloads() -> None:
    """Legacy selector payloads with node references should still restore."""
    objective = SingleAgentMaxObjective()
    frontier = _node(10, 1, score=5.0)
    tree = _tree(
        _node(0, 0, opened=True),
        frontier,
        objective=objective,
    )
    payload = SimpleNamespace(
        type="linoo",
        version=1,
        depth_stats=[
            LinooDepthStatsCheckpointPayload(
                depth=0,
                total_nodes=1,
                opened_count=1,
                frontier_count=0,
                terminal_count=0,
                exact_count=0,
                uncached_terminal_candidates=0,
                non_openable_count=0,
            ),
            LinooDepthStatsCheckpointPayload(
                depth=1,
                total_nodes=1,
                opened_count=0,
                frontier_count=1,
                terminal_count=0,
                exact_count=0,
                uncached_terminal_candidates=0,
                non_openable_count=0,
            ),
        ],
        node_states=[
            SimpleNamespace(node=frontier, depth=1, status="frontier"),
        ],
        candidates_by_depth=[
            SimpleNamespace(
                depth=1,
                candidates=[
                    SimpleNamespace(
                        node=frontier,
                        depth=1,
                        priority=5.0,
                        version=1,
                    )
                ],
            )
        ],
        last_selected_node=frontier,
    )

    restored_selector = _selector()
    restored = restored_selector.restore_from_checkpoint_payload(
        tree=tree,
        objective=objective,
        payload=payload,
    )
    instructions = restored_selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )

    assert restored is True
    assert _selected_node_id(instructions) == 10
    assert restored_selector.latest_selection_report is not None
    assert restored_selector.latest_selection_report.state_rebuilt is False
    assert restored_selector._last_selected_node_id == 10


def test_linoo_checkpoint_restore_rejects_changed_classification() -> None:
    """Payload classifications must match the current live tree."""
    objective = SingleAgentMaxObjective()
    first = _node(10, 1, score=5.0)
    tree = _tree(
        _node(0, 0, opened=True),
        first,
        objective=objective,
    )
    selector = _selector()
    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )
    payload = selector.build_checkpoint_payload(objective)
    assert payload is not None

    first.all_branches_generated = True
    restored_selector = _selector()
    restored = restored_selector.restore_from_checkpoint_payload(
        tree=tree,
        objective=objective,
        payload=payload,
    )

    assert restored is False


def test_linoo_refresh_state_for_checkpoint_updates_initialized_cache_only() -> None:
    """Public checkpoint refresh should update only an initialized Linoo cache."""
    objective = SingleAgentMaxObjective()
    root = _node(0, 0, opened=True)
    first = _node(10, 1, score=5.0)
    second = _node(11, 1, score=4.0)
    tree = _tree(root, first, second, objective=objective)
    selector = _selector()

    selector.refresh_state_for_checkpoint(
        tree=tree,
        objective=objective,
        latest_tree_expansions=TreeExpansions(),
    )

    assert selector.build_checkpoint_payload(objective) is None

    selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=TreeExpansions(),
    )
    child = _node(20, 2, score=100.0)
    first.all_branches_generated = True
    tree.descendants.setdefault(2, {})[child.id] = child
    tree.nodes_count += 1
    expansions: TreeExpansions[_FakeNode] = TreeExpansions()
    expansions.record_creation(
        TreeExpansion(
            child_node=child,
            parent_node=first,
            state_modifications=None,
            creation_child_node=True,
            branch_key=0,
        )
    )

    selector.refresh_state_for_checkpoint(
        tree=tree,
        objective=objective,
        latest_tree_expansions=expansions,
    )
    payload = selector.build_checkpoint_payload(objective)

    assert payload is not None
    assert {node_state.node_id for node_state in payload.node_states} == {
        11,
        20,
    }
    assert selector._node_state_or_none(10) is None


def test_linoo_zipf_sampling_respects_rank_intervals() -> None:
    """Zipf sampling should follow rank order over valid selected-depth candidates."""
    tree = _tree(
        _node(0, 0, opened=True),
        _node(10, 1, score=1.0),
        _node(11, 1, score=5.0),
        _node(12, 1, score=3.0),
    )

    first_selector = _selector(random_values=[0.0])
    second_selector = _selector(random_values=[0.7])
    third_selector = _selector(random_values=[0.999999])

    first_instructions = first_selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )
    second_instructions = second_selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )
    third_instructions = third_selector.choose_node_and_branch_to_open(
        tree=tree,
        latest_tree_expansions=SimpleNamespace(),
    )

    assert _selected_node_id(first_instructions) == 11
    assert _selected_node_id(second_instructions) == 12
    assert _selected_node_id(third_instructions) == 10
    assert first_selector.latest_selection_report is not None
    assert first_selector.latest_selection_report.selected_node_rank == 1
    assert second_selector.latest_selection_report is not None
    assert second_selector.latest_selection_report.selected_node_rank == 2
    assert third_selector.latest_selection_report is not None
    assert third_selector.latest_selection_report.selected_node_rank == 3


def test_linoo_breaks_equal_depth_indices_by_smaller_depth() -> None:
    """Equal depth indices should deterministically prefer the shallower depth."""
    selector = _selector(depth_selection_policy="opened_count_depth_index")
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
    """Equal priorities should rank the smallest node id first."""
    selector = _selector()
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
    report = selector.latest_selection_report
    assert report is not None
    assert report.selected_node_rank == 1


def test_linoo_raises_when_direct_value_is_unavailable() -> None:
    """Linoo should fail fast when a frontier node lacks direct_value access."""
    selector = _selector()
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
    selector = _selector()
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
        args=LinooArgs(
            type=NodeSelectorType.LINOO,
            depth_selection_policy="opened_count_depth_index",
        ),
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
    assert selector.depth_selection_policy == "opened_count_depth_index"
    assert isinstance(composed_selector.base, Linoo)
    assert composed_selector.base.depth_selection_policy == "inverse_depth"
