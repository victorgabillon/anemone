"""Tests for debug-layer search event observers."""

# ruff: noqa: D101, D102, D103

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace
from typing import Any

from anemone.debug import (
    BackupFinished,
    BackupStarted,
    ChildLinked,
    DirectValueAssigned,
    NodeOpeningPlanned,
    NodeSelected,
    ObservableAlgorithmNodeTreeManager,
    ObservableDirectEvaluator,
    ObservableNodeSelector,
    ObservableTreeExploration,
    ObservableUpdater,
    SearchDebugEvent,
    SearchIterationCompleted,
    SearchIterationStarted,
)


@dataclass
class RecordingDebugSink:
    events: list[SearchDebugEvent] = field(default_factory=list)

    def emit(self, event: SearchDebugEvent) -> None:
        self.events.append(event)


@dataclass
class _FakeValue:
    score: float
    certainty: str | None = None
    over_event: object | None = None


@dataclass
class _FakeTreeEvaluation:
    direct_value: _FakeValue | None = None
    backed_up_value: _FakeValue | None = None
    best_branch_sequence: list[int] = field(default_factory=list)
    over_event: object | None = None
    current_value: _FakeValue = field(default_factory=lambda: _FakeValue(score=0.0))

    def has_exact_value(self) -> bool:
        return False

    def is_terminal(self) -> bool:
        return False

    def get_value(self) -> _FakeValue:
        return self.backed_up_value or self.direct_value or self.current_value

    def get_score(self) -> float:
        return self.get_value().score


@dataclass(eq=False)
class _FakeNode:
    id: int
    tree_depth: int
    state: Any
    tree_evaluation: _FakeTreeEvaluation = field(default_factory=_FakeTreeEvaluation)
    branches_children: dict[int, _FakeNode | None] = field(default_factory=dict)
    parent_nodes: dict[_FakeNode, int] = field(default_factory=dict)

    def add_parent(self, branch_key: int, new_parent_node: _FakeNode) -> None:
        self.parent_nodes[new_parent_node] = branch_key


@dataclass
class _FakeOpeningInstruction:
    node_to_open: _FakeNode
    branch: int


class _FakeOpeningInstructions:
    def __init__(self, instructions: list[_FakeOpeningInstruction]) -> None:
        self._instructions = instructions

    def values(self) -> tuple[_FakeOpeningInstruction, ...]:
        return tuple(self._instructions)


class _FakeSelector:
    def __init__(self, opening_instructions: _FakeOpeningInstructions) -> None:
        self._opening_instructions = opening_instructions

    def choose_node_and_branch_to_open(
        self,
        *,
        tree: Any,
        latest_tree_expansions: Any,
    ) -> _FakeOpeningInstructions:
        del tree, latest_tree_expansions
        return self._opening_instructions


class _FakeListSelector:
    def __init__(self, instructions: list[_FakeOpeningInstruction]) -> None:
        self._instructions = instructions

    def choose_node_and_branch_to_open(
        self,
        *,
        tree: Any,
        latest_tree_expansions: Any,
    ) -> list[_FakeOpeningInstruction]:
        del tree, latest_tree_expansions
        return list(self._instructions)


class _FakeEvaluationQueries:
    def __init__(self) -> None:
        self.over_nodes: list[_FakeNode] = []
        self.not_over_nodes: list[_FakeNode] = []

    def clear_queries(self) -> None:
        self.over_nodes = []
        self.not_over_nodes = []


class _FakeDirectEvaluator:
    def add_evaluation_query(self, node: _FakeNode, evaluation_queries: Any) -> None:
        node.tree_evaluation.direct_value = _FakeValue(score=1.0, certainty="terminal")
        evaluation_queries.over_nodes.append(node)

    def evaluate_all_queried_nodes(self, evaluation_queries: Any) -> None:
        for node in evaluation_queries.not_over_nodes:
            node.tree_evaluation.direct_value = _FakeValue(
                score=node.state.base_score,
                certainty="estimate",
            )
        evaluation_queries.clear_queries()

    def evaluate_all_not_over(self, not_over_nodes: list[_FakeNode]) -> None:
        for node in not_over_nodes:
            node.tree_evaluation.direct_value = _FakeValue(
                score=node.state.base_score,
                certainty="estimate",
            )


class _FakeNoChangeDirectEvaluator:
    def evaluate_all_not_over(self, not_over_nodes: list[_FakeNode]) -> None:
        del not_over_nodes


class _FakeUpdateInstructionsBatch:
    def __init__(self, items: list[tuple[_FakeNode, Any]] | None = None) -> None:
        self.one_node_instructions = {} if items is None else dict(enumerate(items))

    def __bool__(self) -> bool:
        return bool(self.one_node_instructions)

    def pop_item(self) -> tuple[_FakeNode, Any]:
        _, item = self.one_node_instructions.popitem()
        return item

    def add_updates_towards_one_parent_node(
        self,
        parent_node: _FakeNode,
        update_from_child_node: Any,
    ) -> None:
        key = len(self.one_node_instructions)
        self.one_node_instructions[key] = (parent_node, update_from_child_node)

    def add_update_from_one_child_node(
        self,
        update_from_child_node: Any,
        parent_node: _FakeNode,
        branch_from_parent: int,
    ) -> None:
        del branch_from_parent
        self.add_updates_towards_one_parent_node(parent_node, update_from_child_node)


class _FakeUpdater:
    def __init__(self) -> None:
        self.generate_result = _FakeUpdateInstructionsBatch()

    def create_update_instructions_after_node_birth(self, new_node: _FakeNode) -> Any:
        del new_node
        return SimpleNamespace()

    def generate_update_instructions(
        self, tree_expansions: Any
    ) -> _FakeUpdateInstructionsBatch:
        del tree_expansions
        return self.generate_result

    def perform_updates(
        self, node_to_update: _FakeNode, update_instructions: Any
    ) -> Any:
        del update_instructions
        node_to_update.tree_evaluation.backed_up_value = _FakeValue(score=2.5)
        node_to_update.tree_evaluation.best_branch_sequence = [7]
        node_to_update.tree_evaluation.over_event = "over"
        return SimpleNamespace()


class _FakeNoOpUpdater:
    def perform_updates(
        self, node_to_update: _FakeNode, update_instructions: Any
    ) -> Any:
        del node_to_update, update_instructions
        return SimpleNamespace(marker="no-op")


@dataclass
class _FakeTreeExpansion:
    child_node: _FakeNode
    parent_node: _FakeNode | None
    state_modifications: Any
    creation_child_node: bool
    branch_key: int | None


class _FakeObservedOpenManager:
    def __init__(self) -> None:
        self.dynamics = SimpleNamespace(
            action_name=lambda state, branch: f"move:{branch}"
        )
        self.was_called = False
        self.received_tree: Any | None = None
        self.received_opening_instructions: Any | None = None
        self.result: tuple[_FakeTreeExpansion, ...] = ()

    def expand_instructions(self, tree: Any, opening_instructions: Any) -> Any:
        self.was_called = True
        self.received_tree = tree
        self.received_opening_instructions = opening_instructions

        expansions: list[_FakeTreeExpansion] = []
        for opening_instruction in opening_instructions.values():
            child = _FakeNode(
                id=tree.nodes_count,
                tree_depth=opening_instruction.node_to_open.tree_depth + 1,
                state=SimpleNamespace(tag=f"child-{opening_instruction.branch}"),
                parent_nodes={
                    opening_instruction.node_to_open: opening_instruction.branch
                },
            )
            opening_instruction.node_to_open.branches_children[
                opening_instruction.branch
            ] = child
            tree.nodes_count += 1
            expansions.append(
                _FakeTreeExpansion(
                    child_node=child,
                    parent_node=opening_instruction.node_to_open,
                    state_modifications=None,
                    creation_child_node=True,
                    branch_key=opening_instruction.branch,
                )
            )

        self.result = tuple(expansions)
        return self.result

    def evaluate_expansions(self, tree_expansions: Any) -> None:
        del tree_expansions

    def update_indices(self, tree: Any) -> None:
        del tree

    def print_best_line(self, tree: Any) -> None:
        del tree


class _FakeObservedOpenManagerWithEvaluation(_FakeObservedOpenManager):
    def __init__(self) -> None:
        super().__init__()
        self.node_evaluator = _FakeDirectEvaluator()
        self.evaluation_queries = _FakeEvaluationQueries()

    def evaluate_expansions(self, tree_expansions: Any) -> None:
        for tree_expansion in tree_expansions:
            child = tree_expansion.child_node
            self.node_evaluator.add_evaluation_query(child, self.evaluation_queries)

        self.node_evaluator.evaluate_all_queried_nodes(self.evaluation_queries)


class _FakeObservedUpdateManager:
    def __init__(self) -> None:
        self.dynamics = SimpleNamespace(
            action_name=lambda state, branch: f"move:{branch}"
        )
        self.was_called = False
        self.received_tree_expansions: Any | None = None

    def update_backward(self, tree_expansions: Any) -> None:
        self.was_called = True
        self.received_tree_expansions = tree_expansions

        for tree_expansion in tree_expansions:
            tree_expansion.child_node.tree_evaluation.backed_up_value = _FakeValue(
                score=1.5
            )
            if tree_expansion.parent_node is None:
                continue

            tree_expansion.parent_node.tree_evaluation.backed_up_value = _FakeValue(
                score=2.5
            )
            tree_expansion.parent_node.tree_evaluation.best_branch_sequence = [
                tree_expansion.branch_key or 0
            ]
            tree_expansion.parent_node.tree_evaluation.over_event = "solved"

    def update_indices(self, tree: Any) -> None:
        del tree

    def print_best_line(self, tree: Any) -> None:
        del tree


class _FakeObservedExplorationManager:
    def __init__(self, root: _FakeNode, child: _FakeNode, branch_key: int) -> None:
        self._root = root
        self._child = child
        self._branch_key = branch_key
        self.dynamics = SimpleNamespace(
            action_name=lambda state, branch: f"move:{branch}"
        )

    def expand_instructions(
        self, tree: Any, opening_instructions: Any
    ) -> tuple[_FakeTreeExpansion, ...]:
        del tree, opening_instructions
        self._root.branches_children[self._branch_key] = self._child
        return (
            _FakeTreeExpansion(
                child_node=self._child,
                parent_node=self._root,
                state_modifications=None,
                creation_child_node=True,
                branch_key=self._branch_key,
            ),
        )

    def evaluate_expansions(self, tree_expansions: Any) -> None:
        del tree_expansions

    def update_backward(self, tree_expansions: Any) -> None:
        del tree_expansions
        self._root.tree_evaluation.best_branch_sequence = [self._branch_key]

    def refresh_exploration_indices(self, tree: Any) -> None:
        del tree

    def print_best_line(self, tree: Any) -> None:
        del tree


@dataclass
class _FakeObservedExploration:
    tree: Any
    tree_manager: Any
    node_selector: Any
    recommend_branch_after_exploration: Any
    stopping_criterion: Any
    notify_percent_function: Any
    explore_calls: list[Random] = field(default_factory=list)
    result: Any = field(
        default_factory=lambda: SimpleNamespace(marker="explore-result")
    )

    def print_info_during_branch_computation(self, random_generator: Random) -> None:
        del random_generator

    def explore(self, random_generator: Random) -> Any:
        self.explore_calls.append(random_generator)

        if self.stopping_criterion.should_we_continue(tree=self.tree):
            opening_instructions = self.node_selector.choose_node_and_branch_to_open(
                tree=self.tree,
                latest_tree_expansions=SimpleNamespace(),
            )
            opening_instructions_subset = (
                self.stopping_criterion.respectful_opening_instructions(
                    opening_instructions=opening_instructions,
                    tree=self.tree,
                )
            )
            tree_expansions = self.tree_manager.expand_instructions(
                tree=self.tree,
                opening_instructions=opening_instructions_subset,
            )
            self.tree_manager.evaluate_expansions(tree_expansions=tree_expansions)
            self.tree_manager.update_backward(tree_expansions=tree_expansions)
            self.tree_manager.refresh_exploration_indices(tree=self.tree)

        return self.result


class _FakeStoppingCriterion:
    def __init__(self) -> None:
        self._checks = 0

    def should_we_continue(self, tree: Any) -> bool:
        del tree
        self._checks += 1
        return self._checks == 1

    def respectful_opening_instructions(
        self,
        *,
        opening_instructions: _FakeOpeningInstructions,
        tree: Any,
    ) -> _FakeOpeningInstructions:
        del tree
        return opening_instructions

    def notify_percent_progress(self, tree: Any, notify_percent_function: Any) -> None:
        del tree, notify_percent_function


class _FakeRecommender:
    def policy(self, root_node: _FakeNode) -> object:
        del root_node
        return object()

    def sample(self, policy: object, random_generator: Random) -> int:
        del policy, random_generator
        return 0


def test_recording_debug_sink_collects_events() -> None:
    sink = RecordingDebugSink()

    sink.emit(SearchIterationStarted(iteration_index=2))

    assert sink.events == [SearchIterationStarted(iteration_index=2)]


def test_observable_node_selector_emits_node_selected() -> None:
    node = _FakeNode(id=1, tree_depth=0, state=SimpleNamespace(tag="root"))
    selector = ObservableNodeSelector(
        _FakeSelector(_FakeOpeningInstructions([_FakeOpeningInstruction(node, 0)])),
        debug_sink := RecordingDebugSink(),
    )

    opening_instructions = selector.choose_node_and_branch_to_open(
        tree=SimpleNamespace(root_node=node),
        latest_tree_expansions=SimpleNamespace(),
    )

    assert tuple(opening_instructions.values()) == (_FakeOpeningInstruction(node, 0),)
    assert debug_sink.events == [NodeSelected(node_id="1")]


def test_observable_node_selector_supports_iterable_instruction_containers() -> None:
    node = _FakeNode(id=3, tree_depth=0, state=SimpleNamespace(tag="root"))
    selector = ObservableNodeSelector(
        _FakeListSelector(
            [
                _FakeOpeningInstruction(node, 0),
                _FakeOpeningInstruction(node, 1),
            ]
        ),
        debug_sink := RecordingDebugSink(),
    )

    opening_instructions = selector.choose_node_and_branch_to_open(
        tree=SimpleNamespace(root_node=node),
        latest_tree_expansions=SimpleNamespace(),
    )

    assert isinstance(opening_instructions, list)
    assert debug_sink.events == [NodeSelected(node_id="3")]


def test_observable_direct_evaluator_emits_direct_value_assigned() -> None:
    node = _FakeNode(
        id=4,
        tree_depth=0,
        state=SimpleNamespace(base_score=0.75),
    )
    evaluator = ObservableDirectEvaluator(
        _FakeDirectEvaluator(),
        debug_sink := RecordingDebugSink(),
    )

    evaluator.evaluate_all_not_over([node])

    assert debug_sink.events == [
        DirectValueAssigned(
            node_id="4",
            value_repr="score=0.75, certainty=estimate",
        )
    ]


def test_observable_direct_evaluator_emits_events_for_queried_batches() -> None:
    first = _FakeNode(id=5, tree_depth=0, state=SimpleNamespace(base_score=0.3))
    second = _FakeNode(id=6, tree_depth=0, state=SimpleNamespace(base_score=0.6))
    evaluation_queries = _FakeEvaluationQueries()
    evaluation_queries.not_over_nodes.extend([first, second])
    evaluator = ObservableDirectEvaluator(
        _FakeDirectEvaluator(),
        debug_sink := RecordingDebugSink(),
    )

    evaluator.evaluate_all_queried_nodes(evaluation_queries)

    assert debug_sink.events == [
        DirectValueAssigned(
            node_id="5",
            value_repr="score=0.3, certainty=estimate",
        ),
        DirectValueAssigned(
            node_id="6",
            value_repr="score=0.6, certainty=estimate",
        ),
    ]


def test_observable_direct_evaluator_skips_unchanged_values() -> None:
    node = _FakeNode(
        id=7,
        tree_depth=0,
        state=SimpleNamespace(base_score=0.75),
        tree_evaluation=_FakeTreeEvaluation(
            direct_value=_FakeValue(score=0.75, certainty="estimate")
        ),
    )
    evaluator = ObservableDirectEvaluator(
        _FakeNoChangeDirectEvaluator(),
        debug_sink := RecordingDebugSink(),
    )

    evaluator.evaluate_all_not_over([node])

    assert debug_sink.events == []


def test_observable_updater_emits_backup_events() -> None:
    node = _FakeNode(id=9, tree_depth=1, state=SimpleNamespace(tag="n"))
    updater = ObservableUpdater(_FakeUpdater(), debug_sink := RecordingDebugSink())

    updater.perform_updates(node_to_update=node, update_instructions=SimpleNamespace())

    assert debug_sink.events == [
        BackupStarted(node_id="9"),
        BackupFinished(
            node_id="9",
            value_changed=True,
            pv_changed=True,
            over_changed=True,
        ),
    ]


def test_observable_updater_keeps_flags_false_when_state_is_unchanged() -> None:
    node = _FakeNode(id=10, tree_depth=1, state=SimpleNamespace(tag="n"))
    updater = ObservableUpdater(_FakeNoOpUpdater(), debug_sink := RecordingDebugSink())

    result = updater.perform_updates(
        node_to_update=node,
        update_instructions=SimpleNamespace(),
    )

    assert result == SimpleNamespace(marker="no-op")
    assert debug_sink.events == [
        BackupStarted(node_id="10"),
        BackupFinished(
            node_id="10",
            value_changed=False,
            pv_changed=False,
            over_changed=False,
        ),
    ]


def test_observable_tree_manager_expand_instructions_delegates_and_emits_events() -> (
    None
):
    root = _FakeNode(id=1, tree_depth=0, state=SimpleNamespace(tag="root"))
    tree = SimpleNamespace(nodes_count=2)
    base_manager = _FakeObservedOpenManager()
    manager = ObservableAlgorithmNodeTreeManager(
        base_manager,
        debug_sink := RecordingDebugSink(),
    )

    opening_instructions = _FakeOpeningInstructions([_FakeOpeningInstruction(root, 7)])
    result = manager.expand_instructions(
        tree=tree,
        opening_instructions=opening_instructions,
    )

    assert base_manager.was_called is True
    assert base_manager.received_tree is tree
    assert base_manager.received_opening_instructions is opening_instructions
    assert result is base_manager.result
    assert debug_sink.events == [
        NodeOpeningPlanned(node_id="1", branch_count=1),
        ChildLinked(
            parent_id="1",
            child_id="2",
            branch_key="7",
            was_already_present=False,
        ),
    ]


def test_observable_tree_manager_wraps_direct_evaluator_for_evaluation_phase() -> None:
    root = _FakeNode(id=1, tree_depth=0, state=SimpleNamespace(tag="root"))
    tree = SimpleNamespace(nodes_count=2)
    manager = ObservableAlgorithmNodeTreeManager(
        _FakeObservedOpenManagerWithEvaluation(),
        debug_sink := RecordingDebugSink(),
    )

    tree_expansions = manager.expand_instructions(
        tree=tree,
        opening_instructions=_FakeOpeningInstructions(
            [_FakeOpeningInstruction(root, 7)]
        ),
    )
    manager.evaluate_expansions(tree_expansions=tree_expansions)

    assert (
        DirectValueAssigned(
            node_id="2",
            value_repr="score=1.0, certainty=terminal",
        )
        in debug_sink.events
    )
    assert debug_sink.events.index(
        ChildLinked(
            parent_id="1",
            child_id="2",
            branch_key="7",
            was_already_present=False,
        )
    ) < debug_sink.events.index(
        DirectValueAssigned(
            node_id="2",
            value_repr="score=1.0, certainty=terminal",
        )
    )


def test_observable_tree_manager_update_backward_delegates_and_emits_events() -> None:
    root = _FakeNode(id=1, tree_depth=0, state=SimpleNamespace(tag="root"))
    child = _FakeNode(
        id=2,
        tree_depth=1,
        state=SimpleNamespace(tag="child"),
        parent_nodes={root: 5},
    )
    root.branches_children[5] = child

    tree_expansions = (
        _FakeTreeExpansion(
            child_node=child,
            parent_node=root,
            state_modifications=None,
            creation_child_node=True,
            branch_key=5,
        ),
    )
    base_manager = _FakeObservedUpdateManager()
    manager = ObservableAlgorithmNodeTreeManager(
        base_manager,
        debug_sink := RecordingDebugSink(),
    )

    result = manager.update_backward(tree_expansions=tree_expansions)

    assert result is None
    assert base_manager.was_called is True
    assert base_manager.received_tree_expansions is tree_expansions
    assert debug_sink.events == [
        BackupStarted(node_id="2"),
        BackupStarted(node_id="1"),
        BackupFinished(
            node_id="2",
            value_changed=True,
            pv_changed=False,
            over_changed=False,
        ),
        BackupFinished(
            node_id="1",
            value_changed=True,
            pv_changed=True,
            over_changed=True,
        ),
    ]


def test_observable_tree_exploration_emits_iteration_boundaries() -> None:
    root = _FakeNode(id=1, tree_depth=0, state=SimpleNamespace(tag="root"))
    child = _FakeNode(
        id=2,
        tree_depth=1,
        state=SimpleNamespace(tag="child"),
        parent_nodes={root: 4},
    )
    opening_instructions = _FakeOpeningInstructions([_FakeOpeningInstruction(root, 4)])
    base_exploration = _FakeObservedExploration(
        tree=SimpleNamespace(root_node=root),
        tree_manager=_FakeObservedExplorationManager(root, child, 4),
        node_selector=_FakeSelector(opening_instructions),
        recommend_branch_after_exploration=_FakeRecommender(),
        stopping_criterion=_FakeStoppingCriterion(),
        notify_percent_function=lambda progress: None,
    )
    exploration = ObservableTreeExploration.from_tree_exploration(
        base_exploration,
        debug_sink=(debug_sink := RecordingDebugSink()),
    )

    result = exploration.explore(random_generator=Random(0))

    assert result is base_exploration.result
    assert len(base_exploration.explore_calls) == 1
    assert SearchIterationStarted(iteration_index=1) in debug_sink.events
    assert SearchIterationCompleted(iteration_index=1) in debug_sink.events
