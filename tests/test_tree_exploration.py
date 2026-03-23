"""Focused tests for ``TreeExploration`` iteration orchestration."""

# ruff: noqa: D103

from dataclasses import dataclass, field
from random import Random
from types import SimpleNamespace
from typing import Any

from valanga.evaluations import Certainty, Value

from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
)
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
    def __init__(self) -> None:
        self.calls: list[Any] = []
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
    tree = SimpleNamespace(root_node=root)
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

    exploration.step()

    assert [expansion.child_node for expansion in selector.calls[0][1]] == [root]
    assert stopping.respect_calls == [(selector.instructions, tree)]
    assert manager.calls == [
        "expand",
        "evaluate",
        "propagate_value",
        "propagate_depth",
        "refresh_indices",
    ]


def test_tree_exploration_explore_routes_iterations_through_step() -> None:
    root = _fake_root()
    tree = SimpleNamespace(root_node=root)
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
