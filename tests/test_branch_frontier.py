"""Focused tests for the extracted branch-frontier concept."""

from dataclasses import dataclass, field
from enum import Enum
from random import Random
from types import SimpleNamespace
from typing import Any, cast

from valanga import Color
from valanga.evaluations import Certainty, Value

from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy
from anemone.node_evaluation.common.branch_frontier import BranchFrontierState
from anemone.node_evaluation.direct.node_direct_evaluator import EvaluationQueries
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.tree.decision_ordering import BranchOrderingKey
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.node_selector.branch_explorer import SamplingPriorities
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.recurzipf.recur_zipf_base import (
    RecurZipfBase,
    RecurZipfBaseArgs,
)
from anemone.tree_manager.algorithm_node_tree_manager import AlgorithmNodeTreeManager
from anemone.tree_manager.tree_expander import TreeExpansion


class _SoloRole(Enum):
    SOLO = "solo"


@dataclass(slots=True)
class _FakeFrontierEvaluation:
    frontier: BranchFrontierState = field(default_factory=BranchFrontierState)
    ordered_branches: list[Any] = field(default_factory=list)
    opened_branches: list[Any] = field(default_factory=list)

    def on_branch_opened(self, branch: Any) -> None:
        self.opened_branches.append(branch)
        self.frontier.on_branch_opened(branch)

    def sync_branch_frontier(self, branches_to_refresh: set[Any]) -> None:
        del branches_to_refresh

    def has_frontier_branches(self) -> bool:
        return self.frontier.has_frontier_branches()

    def frontier_branches_in_order(self) -> list[Any]:
        return self.frontier.ordered_frontier_branches(self.ordered_branches)

    def has_exact_value(self) -> bool:
        return False


class _FakeTreeManager:
    def open_tree_expansion_from_branch(
        self,
        tree: object,
        parent_node: object,
        branch: object,
    ) -> TreeExpansion[Any]:
        del tree
        return TreeExpansion(
            child_node=SimpleNamespace(id=2),
            parent_node=parent_node,
            state_modifications=None,
            creation_child_node=True,
            branch_key=branch,
        )


class _FakeOpeningInstructor:
    def all_branches_to_open(self, node_to_open: object) -> list[str]:
        del node_to_open
        return ["open-a", "open-b"]


class _FakeOverEvent:
    def __init__(self, *, winner: Color | None = None, draw: bool = False) -> None:
        self.winner = winner
        self.draw = draw

    def is_over(self) -> bool:
        return self.draw or self.winner is not None

    def is_draw(self) -> bool:
        return self.draw

    def is_win_for(self, role: Color) -> bool:
        return self.winner == role

    def is_loss_for(self, role: Color) -> bool:
        return self.winner is not None and self.winner != role


def _minimax_leaf(node_id: int, value: Value) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=tree_node)
    evaluation.direct_value = value
    evaluation.minmax_value = value
    return SimpleNamespace(id=node_id, tree_node=tree_node, tree_evaluation=evaluation)


def _single_leaf(node_id: int, value: Value) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=SimpleNamespace(turn=_SoloRole.SOLO, phase="single-agent"),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMaxEvaluation(tree_node=tree_node)
    evaluation.direct_value = value
    evaluation.backed_up_value = value
    return SimpleNamespace(id=node_id, tree_node=tree_node, tree_evaluation=evaluation)


def test_tree_manager_notifies_branch_frontier_when_opening_a_branch() -> None:
    tree_manager = AlgorithmNodeTreeManager(
        tree_manager=cast("Any", _FakeTreeManager()),
        algorithm_tree_node_factory=cast("Any", object()),
        evaluation_queries=EvaluationQueries(),
        node_evaluator=None,
        index_manager=cast("Any", object()),
    )
    parent_evaluation = _FakeFrontierEvaluation(ordered_branches=["left"])
    parent_node = SimpleNamespace(tree_evaluation=parent_evaluation)

    tree_manager.open_tree_expansion_from_branch(
        tree=cast("Any", object()),
        parent_node=parent_node,
        branch="left",
    )

    assert parent_evaluation.opened_branches == ["left"]
    assert parent_evaluation.frontier_branches_in_order() == ["left"]


def test_recurzipf_uses_branch_frontier_capability_for_tree_descent() -> None:
    leaf = SimpleNamespace(
        id=2,
        branches_children={},
        tree_evaluation=_FakeFrontierEvaluation(),
    )
    root_evaluation = _FakeFrontierEvaluation(ordered_branches=["left"])
    root_evaluation.on_branch_opened("left")
    root = SimpleNamespace(
        id=1,
        branches_children={"left": leaf},
        tree_evaluation=root_evaluation,
    )
    selector = RecurZipfBase(
        args=RecurZipfBaseArgs(
            type=NodeSelectorType.RECUR_ZIPF_BASE,
            branch_explorer_priority=SamplingPriorities.NO_PRIORITY,
        ),
        random_generator=Random(0),
        opening_instructor=cast("Any", _FakeOpeningInstructor()),
    )

    instructions = selector.choose_node_and_branch_to_open(
        tree=cast("Any", SimpleNamespace(root_node=root)),
        latest_tree_expansions=cast("Any", object()),
    )

    assert len(instructions) == 2
    assert all(
        instruction.node_to_open is leaf for instruction in instructions.values()
    )


def test_minimax_frontier_follows_branch_order_and_clears_for_exact_nodes() -> None:
    low_child = _minimax_leaf(
        1,
        Value(score=0.1, certainty=Certainty.ESTIMATE),
    )
    high_child = _minimax_leaf(
        2,
        Value(score=0.9, certainty=Certainty.ESTIMATE),
    )
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={"low": low_child, "high": high_child},
        all_branches_generated=False,
    )
    parent = NodeMinmaxEvaluation(
        tree_node=parent_tree_node,
        backup_policy=ExplicitMinimaxBackupPolicy(),
    )
    parent.direct_value = Value(score=0.0, certainty=Certainty.ESTIMATE)
    parent.minmax_value = parent.direct_value

    parent.backup_from_children(
        branches_with_updated_value={"low", "high"},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.frontier_branches_in_order() == ["high", "low"]

    winning_child = _minimax_leaf(
        3,
        Value(
            score=1.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(winner=Color.WHITE),
        ),
    )
    live_child = _minimax_leaf(
        4,
        Value(score=0.2, certainty=Certainty.ESTIMATE),
    )
    exact_parent_tree_node = SimpleNamespace(
        id=5,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={"win": winning_child, "live": live_child},
        all_branches_generated=False,
    )
    exact_parent = NodeMinmaxEvaluation(
        tree_node=exact_parent_tree_node,
        backup_policy=ExplicitMinimaxBackupPolicy(),
    )
    exact_parent.direct_value = Value(score=0.0, certainty=Certainty.ESTIMATE)
    exact_parent.minmax_value = exact_parent.direct_value

    exact_parent.backup_from_children(
        branches_with_updated_value={"win", "live"},
        branches_with_updated_best_branch_seq=set(),
    )

    assert exact_parent.has_exact_value()
    assert not exact_parent.has_frontier_branches()
    assert exact_parent.frontier_branches_in_order() == []


def test_single_agent_frontier_tracks_only_unresolved_children() -> None:
    exact_child = _single_leaf(
        1,
        Value(
            score=2.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(draw=True),
        ),
    )
    live_child = _single_leaf(
        2,
        Value(score=5.0, certainty=Certainty.ESTIMATE),
    )
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=_SoloRole.SOLO, phase="single-agent"),
        branches_children={0: exact_child, 1: live_child},
        all_branches_generated=False,
    )
    parent = NodeMaxEvaluation(
        tree_node=parent_tree_node,
        backup_policy=ExplicitMaxBackupPolicy(),
    )
    parent.direct_value = Value(score=0.0, certainty=Certainty.ESTIMATE)
    parent.on_branch_opened(0)
    parent.on_branch_opened(1)

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.has_frontier_branches()
    assert parent.frontier_branches_in_order() == [1]


def test_single_agent_frontier_uses_explicit_ordering_cache_updates() -> None:
    low_child = _single_leaf(
        1,
        Value(score=0.1, certainty=Certainty.ESTIMATE),
    )
    high_child = _single_leaf(
        2,
        Value(score=0.9, certainty=Certainty.ESTIMATE),
    )
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=_SoloRole.SOLO, phase="single-agent"),
        branches_children={0: low_child, 1: high_child},
        all_branches_generated=False,
    )
    parent = NodeMaxEvaluation(
        tree_node=parent_tree_node,
        backup_policy=ExplicitMaxBackupPolicy(),
    )
    parent.direct_value = Value(score=0.0, certainty=Certainty.ESTIMATE)
    parent.on_branch_opened(0)
    parent.on_branch_opened(1)

    assert parent.decision_ordering.branch_ordering_keys == {}
    assert parent.frontier_branches_in_order() == [0, 1]
    assert parent.decision_ordering.branch_ordering_keys == {}

    parent.update_branches_values({0, 1})

    assert parent.frontier_branches_in_order() == [1, 0]
    assert parent.decision_ordering.branch_ordering_keys == {
        0: BranchOrderingKey(
            primary_score=0.1,
            tactical_tiebreak=0,
            stable_tiebreak_id=1,
        ),
        1: BranchOrderingKey(
            primary_score=0.9,
            tactical_tiebreak=0,
            stable_tiebreak_id=2,
        ),
    }
