"""Focused tests for ``TreeExploration`` iteration orchestration."""

# ruff: noqa: D103

from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace
from typing import Any

from valanga.evaluations import Certainty, Value

from anemone.node_selector.linoo import Linoo
from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
)
from anemone.objectives import SingleAgentMaxObjective
from anemone.tree_exploration import TreeExploration
from anemone.tree_manager.tree_expander import TreeExpansion, TreeExpansions


def _fake_root() -> Any:
    tree_evaluation = SimpleNamespace(
        has_exact_value=lambda: False,
        best_branch_sequence=[],
        get_score=lambda: 0.0,
        get_value=lambda: Value(score=0.0, certainty=Certainty.ESTIMATE),
        print_branch_ordering=lambda dynamics: None,
    )
    state = SimpleNamespace()
    return SimpleNamespace(
        id=0,
        state=state,
        tree_depth=0,
        branches_children={},
        tree_evaluation=tree_evaluation,
    )


class _SelectorSpy:
    def __init__(self, selection_report: object | None = None) -> None:
        self.calls: list[Any] = []
        self._selection_report = selection_report
        self.latest_selection_report: object | None = None
        self.instructions = OpeningInstructions(
            {
                (0, 1): OpeningInstruction(
                    node_to_open=SimpleNamespace(
                        id=0,
                        tree_depth=0,
                        state=SimpleNamespace(),
                    ),
                    branch=1,
                )
            }
        )

    def choose_node_and_branch_to_open(
        self,
        *,
        tree: Any,
        latest_tree_expansions: TreeExpansions[Any],
    ) -> OpeningInstructions[Any]:
        self.calls.append((tree, list(latest_tree_expansions)))
        self.latest_selection_report = self._selection_report
        return self.instructions


class _StoppingCriterionSpy:
    def __init__(self) -> None:
        self.respect_calls: list[Any] = []
        self.continue_checks = 0

    def should_we_continue(self, *, tree: Any) -> bool:
        self.continue_checks += 1
        return self.continue_checks == 1

    def respectful_opening_instructions(
        self,
        *,
        opening_instructions: OpeningInstructions[Any],
        tree: Any,
    ) -> OpeningInstructions[Any]:
        self.respect_calls.append((opening_instructions, tree))
        return opening_instructions

    def notify_percent_progress(
        self,
        *,
        tree: Any,
        notify_percent_function: Any,
    ) -> None:
        del tree, notify_percent_function

    def get_string_of_progress(self, tree: Any) -> str:
        del tree
        return "progress"


@dataclass
class _ManagerSpy:
    calls: list[str] = field(default_factory=list)
    last_expansions: TreeExpansions[Any] | None = None
    dynamics: Any = field(
        default_factory=lambda: SimpleNamespace(
            action_name=lambda state, branch: f"move:{branch}"
        )
    )

    def expand_instructions(
        self,
        *,
        tree: Any,
        opening_instructions: Any,
    ) -> TreeExpansions[Any]:
        del tree, opening_instructions
        self.calls.append("expand")
        expansions: TreeExpansions[Any] = TreeExpansions()
        expansions.add_creation(
            TreeExpansion(
                child_node=SimpleNamespace(id=1),
                parent_node=SimpleNamespace(id=0),
                state_modifications=None,
                creation_child_node=True,
                branch_key=1,
            )
        )
        self.last_expansions = expansions
        return expansions

    def evaluate_expansions(
        self,
        *,
        tree_expansions: TreeExpansions[Any],
    ) -> None:
        assert tree_expansions is self.last_expansions
        self.calls.append("evaluate")

    def update_backward(
        self,
        *,
        tree_expansions: TreeExpansions[Any],
    ) -> None:
        assert tree_expansions is self.last_expansions
        self.calls.append("propagate_value")

    def propagate_depth_index(
        self,
        *,
        tree_expansions: TreeExpansions[Any],
    ) -> None:
        assert tree_expansions is self.last_expansions
        self.calls.append("propagate_depth")

    def refresh_exploration_indices(self, *, tree: Any) -> None:
        del tree
        self.calls.append("refresh_indices")

    def print_best_line(self, *, tree: Any) -> None:
        del tree


class _RecommendSpy:
    def policy(self, root_node: Any) -> object:
        del root_node
        return object()

    def sample(self, policy: object, random_generator: Random) -> int:
        del policy, random_generator
        return 1


def test_tree_exploration_step_owns_iteration_sequence() -> None:
    root = _fake_root()
    tree = SimpleNamespace(root_node=root, nodes_count=1, branch_count=0)
    manager = _ManagerSpy()
    selector = _SelectorSpy()
    stopping = _StoppingCriterionSpy()
    exploration = TreeExploration(
        tree=tree,
        tree_manager=manager,
        node_selector=selector,
        recommend_branch_after_exploration=_RecommendSpy(),
        stopping_criterion=stopping,
        notify_percent_function=lambda progress: None,
    )

    report = exploration.step()

    assert [expansion.child_node for expansion in selector.calls[0][1]] == [root]
    assert stopping.respect_calls == [(selector.instructions, tree)]
    assert manager.calls == [
        "expand",
        "evaluate",
        "propagate_value",
        "propagate_depth",
        "refresh_indices",
    ]
    assert report.select_s is not None and report.select_s >= 0.0
    assert report.limit_s is not None and report.limit_s >= 0.0
    assert report.expand_s is not None and report.expand_s >= 0.0
    assert report.evaluate_s is not None and report.evaluate_s >= 0.0
    assert report.propagate_s is not None and report.propagate_s >= 0.0
    assert report.total_s is not None and report.total_s >= 0.0


def test_tree_exploration_explore_routes_iterations_through_step() -> None:
    root = _fake_root()
    tree = SimpleNamespace(root_node=root, nodes_count=1, branch_count=0)
    manager = _ManagerSpy()
    selector = _SelectorSpy()
    stopping = _StoppingCriterionSpy()
    exploration = TreeExploration(
        tree=tree,
        tree_manager=manager,
        node_selector=selector,
        recommend_branch_after_exploration=_RecommendSpy(),
        stopping_criterion=stopping,
        notify_percent_function=lambda progress: None,
    )

    result = exploration.explore(random_generator=Random(0))

    assert result.tree is tree
    assert manager.calls == [
        "expand",
        "evaluate",
        "propagate_value",
        "propagate_depth",
        "refresh_indices",
    ]


def test_tree_exploration_step_attaches_selector_report_when_available() -> None:
    root = _fake_root()
    tree = SimpleNamespace(root_node=root, nodes_count=1, branch_count=0)
    manager = _ManagerSpy()
    selection_report = SimpleNamespace(depth_rows=(object(), object()))
    selector = _SelectorSpy(selection_report=selection_report)
    stopping = _StoppingCriterionSpy()
    exploration = TreeExploration(
        tree=tree,
        tree_manager=manager,
        node_selector=selector,
        recommend_branch_after_exploration=_RecommendSpy(),
        stopping_criterion=stopping,
        notify_percent_function=lambda progress: None,
    )

    report = exploration.step()

    assert report.selector_report is selection_report
    assert report.selector_report_rows == 2


@dataclass
class _LinooIntegrationEval:
    direct_value: Value | None
    required_objective: object | None = None
    exact: bool = False
    terminal: bool = False

    def has_exact_value(self) -> bool:
        return self.exact

    def is_terminal(self) -> bool:
        return self.terminal

    def get_score(self) -> float:
        assert self.direct_value is not None
        return self.direct_value.score

    def get_value(self) -> Value:
        assert self.direct_value is not None
        return self.direct_value

    def print_branch_ordering(self, dynamics: object) -> None:
        del dynamics


@dataclass
class _LinooIntegrationNode:
    id: int
    tree_depth: int
    tree_evaluation: _LinooIntegrationEval
    all_branches_generated: bool = False
    state: object = field(default_factory=lambda: SimpleNamespace(turn="solo"))
    branches_children: dict[int, object] = field(default_factory=dict)


def _linoo_integration_node(
    node_id: int,
    depth: int,
    score: float,
    *,
    opened: bool = False,
) -> _LinooIntegrationNode:
    return _LinooIntegrationNode(
        id=node_id,
        tree_depth=depth,
        tree_evaluation=_LinooIntegrationEval(
            direct_value=Value(score=score, certainty=Certainty.ESTIMATE)
        ),
        all_branches_generated=opened,
    )


def _linoo_integration_tree(
    *nodes: _LinooIntegrationNode,
) -> SimpleNamespace:
    root = min(nodes, key=lambda node: (node.tree_depth, node.id))
    root.tree_evaluation.required_objective = SingleAgentMaxObjective()
    descendants: dict[int, dict[int, _LinooIntegrationNode]] = {}
    for node in nodes:
        descendants.setdefault(node.tree_depth, {})[node.id] = node
    return SimpleNamespace(
        root_node=root,
        descendants=descendants,
        nodes_count=len(nodes),
        branch_count=0,
        node_depth=lambda node: node.tree_depth - root.tree_depth,
    )


def _scan_linoo_integration_selected_node_id(tree: SimpleNamespace) -> int:
    """Return the selected node id using the old scan-based Linoo policy."""
    depth_state_by_depth: dict[int, tuple[int, list[_LinooIntegrationNode]]] = {}
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
    selected_depth = min(
        frontier_by_depth,
        key=lambda depth: (depth_state_by_depth[depth][0] * (depth + 1), depth),
    )
    objective = tree.root_node.tree_evaluation.required_objective
    return max(
        frontier_by_depth[selected_depth],
        key=lambda node: (
            objective.evaluate_value(node.tree_evaluation.direct_value, node.state),
            -node.id,
        ),
    ).id


class _LinooIntegrationOpeningInstructor:
    def all_branches_to_open(self, node_to_open: object) -> list[int]:
        del node_to_open
        return [0]


class _LinooIntegrationManager:
    dynamics: object = SimpleNamespace(action_name=lambda state, branch: str(branch))
    node_evaluator: object = SimpleNamespace(
        master_state_value_evaluator=None,
        current_evaluator_version=0,
    )

    def expand_instructions(
        self,
        *,
        tree: SimpleNamespace,
        opening_instructions: OpeningInstructions[_LinooIntegrationNode],
    ) -> TreeExpansions[_LinooIntegrationNode]:
        instruction = next(iter(opening_instructions.values()))
        selected_node = instruction.node_to_open
        selected_node.all_branches_generated = True
        expansions: TreeExpansions[_LinooIntegrationNode] = TreeExpansions()
        if selected_node.id == 10:
            child = _linoo_integration_node(20, 2, 100.0)
            tree.descendants.setdefault(child.tree_depth, {})[child.id] = child
            tree.nodes_count += 1
            tree.branch_count += 1
            expansions.record_creation(
                TreeExpansion(
                    child_node=child,
                    parent_node=selected_node,
                    state_modifications=None,
                    creation_child_node=True,
                    branch_key=instruction.branch,
                )
            )
        return expansions

    def evaluate_expansions(
        self,
        *,
        tree_expansions: TreeExpansions[_LinooIntegrationNode],
    ) -> None:
        del tree_expansions

    def update_backward(
        self,
        *,
        tree_expansions: TreeExpansions[_LinooIntegrationNode],
    ) -> None:
        del tree_expansions

    def propagate_depth_index(
        self,
        *,
        tree_expansions: TreeExpansions[_LinooIntegrationNode],
    ) -> None:
        del tree_expansions

    def refresh_exploration_indices(self, *, tree: object) -> None:
        del tree

    def reevaluate_nodes(
        self,
        *,
        tree: object,
        nodes: list[_LinooIntegrationNode],
    ) -> object:
        del tree
        for node in nodes:
            assert node.tree_evaluation.direct_value is not None
            node.tree_evaluation.direct_value = Value(
                score=node.tree_evaluation.direct_value.score + 1.0,
                certainty=Certainty.ESTIMATE,
            )
        return SimpleNamespace(
            reevaluated_count=len(nodes),
            changed_count=len(nodes),
            skipped_terminal_count=0,
        )

    def print_best_line(self, *, tree: object) -> None:
        del tree


class _LinooIntegrationStoppingCriterion:
    def should_we_continue(self, *, tree: object) -> bool:
        del tree
        return True

    def respectful_opening_instructions(
        self,
        *,
        opening_instructions: OpeningInstructions[_LinooIntegrationNode],
        tree: object,
    ) -> OpeningInstructions[_LinooIntegrationNode]:
        del tree
        return opening_instructions

    def notify_percent_progress(
        self,
        *,
        tree: object,
        notify_percent_function: object,
    ) -> None:
        del tree, notify_percent_function

    def get_string_of_progress(self, tree: object) -> str:
        del tree
        return "progress"


def test_linoo_incremental_cache_updates_through_tree_exploration_step() -> None:
    """Real ``TreeExploration.step`` should feed Linoo incremental expansions."""
    root = _linoo_integration_node(0, 0, 0.0, opened=True)
    first = _linoo_integration_node(10, 1, 5.0)
    second = _linoo_integration_node(11, 1, 4.0)
    tree = _linoo_integration_tree(root, first, second)
    selector = Linoo(opening_instructor=_LinooIntegrationOpeningInstructor())
    exploration = TreeExploration(
        tree=tree,
        tree_manager=_LinooIntegrationManager(),
        node_selector=selector,
        recommend_branch_after_exploration=_RecommendSpy(),
        stopping_criterion=_LinooIntegrationStoppingCriterion(),
        notify_percent_function=lambda progress: None,
    )

    assert _scan_linoo_integration_selected_node_id(tree) == 10
    first_report = exploration.step()

    assert first_report.selected_node_id == 10
    assert first_report.selector_report is not None
    assert first_report.selector_report.state_rebuilt is True
    assert _scan_linoo_integration_selected_node_id(tree) == 20

    second_report = exploration.step()

    assert second_report.selected_node_id == 20
    assert second_report.selector_report is not None
    assert second_report.selector_report.state_rebuilt is False
    assert second_report.selector_report.nodes_incrementally_updated == 2
    assert second_report.selector_report.total_nodes_scanned < tree.nodes_count


def test_reevaluation_invalidates_linoo_incremental_cache_before_next_step() -> None:
    """Reevaluation should force Linoo to rebuild before the next selection."""
    root = _linoo_integration_node(0, 0, 0.0, opened=True)
    first = _linoo_integration_node(10, 1, 5.0)
    second = _linoo_integration_node(11, 1, 4.0)
    tree = _linoo_integration_tree(root, first, second)
    selector = Linoo(opening_instructor=_LinooIntegrationOpeningInstructor())
    exploration = TreeExploration(
        tree=tree,
        tree_manager=_LinooIntegrationManager(),
        node_selector=selector,
        recommend_branch_after_exploration=_RecommendSpy(),
        stopping_criterion=_LinooIntegrationStoppingCriterion(),
        notify_percent_function=lambda progress: None,
    )

    first_report = exploration.step()
    assert first_report.selector_report is not None
    assert first_report.selector_report.state_rebuilt is True

    second_report = exploration.step()
    assert second_report.selector_report is not None
    assert second_report.selector_report.state_rebuilt is False

    frontier_node = second
    reevaluation_report = exploration.reevaluate_nodes([frontier_node])
    assert reevaluation_report.reevaluated_count == 1

    third_report = exploration.step()
    assert third_report.selector_report is not None
    assert third_report.selector_report.state_rebuilt is True
