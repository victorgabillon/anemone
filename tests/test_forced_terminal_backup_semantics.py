"""Semantic regression tests for ESTIMATE/FORCED/TERMINAL backup behavior."""

from types import SimpleNamespace
from typing import Any

from valanga import Color
from valanga.evaluations import Certainty, Value

from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)


class _FakeOverEvent:
    def __init__(self, *, winner: Color | None = None, draw: bool = False) -> None:
        self.winner = winner
        self.draw = draw

    def is_over(self) -> bool:
        return self.draw or self.winner is not None

    def is_win(self) -> bool:
        return self.winner is not None

    def is_draw(self) -> bool:
        return self.draw

    def is_winner(self, player: Color) -> bool:
        return self.winner == player


def _single_state() -> Any:
    return SimpleNamespace(phase="single-agent")


def _single_leaf(node_id: int, value: Value) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=_single_state(),
        branches_children={},
        all_branches_generated=True,
    )
    evaluation = NodeMaxEvaluation(tree_node=tree_node)
    evaluation.direct_value = value
    evaluation.backed_up_value = value
    return SimpleNamespace(tree_node=tree_node, tree_evaluation=evaluation)


def _single_parent(*, children: dict[int, Any]) -> NodeMaxEvaluation[Any]:
    tree_node = SimpleNamespace(
        id=0,
        state=_single_state(),
        branches_children=children,
        all_branches_generated=True,
    )
    evaluation = NodeMaxEvaluation(tree_node=tree_node)
    evaluation.direct_value = Value(score=0.0, certainty=Certainty.ESTIMATE)
    return evaluation


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
    return SimpleNamespace(tree_node=tree_node, tree_evaluation=evaluation)


def _minimax_parent(
    *,
    turn: Color,
    children: dict[int | str, Any],
    all_generated: bool,
) -> NodeMinmaxEvaluation[Any, Any]:
    tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=turn),
        branches_children=children,
        all_branches_generated=all_generated,
    )
    evaluation = NodeMinmaxEvaluation(
        tree_node=tree_node,
        backup_policy=ExplicitMinimaxBackupPolicy(),
    )
    evaluation.direct_value = Value(score=0.0, certainty=Certainty.ESTIMATE)
    evaluation.minmax_value = evaluation.direct_value
    return evaluation


def test_single_agent_solved_non_terminal_parent_becomes_forced() -> None:
    child_a = _single_leaf(
        1,
        Value(
            score=8.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(winner=None, draw=True),
        ),
    )
    child_b = _single_leaf(
        2,
        Value(
            score=6.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(winner=None, draw=True),
        ),
    )
    parent = _single_parent(children={0: child_a, 1: child_b})

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert child_a.tree_evaluation.is_terminal()
    assert child_b.tree_evaluation.is_terminal()
    assert parent.backed_up_value is not None
    assert parent.backed_up_value.score == 8.0
    assert parent.backed_up_value.certainty is Certainty.FORCED
    assert not parent.is_terminal()


def test_minimax_partial_winning_child_makes_parent_forced_not_terminal() -> None:
    winning_child = _minimax_leaf(
        1,
        Value(
            score=0.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(winner=Color.WHITE),
        ),
    )
    live_child = _minimax_leaf(
        2,
        Value(score=100.0, certainty=Certainty.ESTIMATE),
    )
    parent = _minimax_parent(
        turn=Color.WHITE,
        children={"win": winning_child, "live": live_child},
        all_generated=False,
    )

    parent.backup_from_children(
        branches_with_updated_value={"win", "live"},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value is not None
    assert parent.minmax_value.certainty is Certainty.FORCED
    assert parent.minmax_value.over_event is not None
    assert parent.minmax_value.over_event.is_winner(Color.WHITE)
    assert not parent.is_terminal()
    assert parent.has_exact_value()
    assert parent.frontier_branches_in_order() == []


def test_minimax_all_terminal_children_make_parent_forced_with_best_outcome() -> None:
    draw_child = _minimax_leaf(
        1,
        Value(
            score=0.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(draw=True),
        ),
    )
    loss_child = _minimax_leaf(
        2,
        Value(
            score=-1.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(winner=Color.BLACK),
        ),
    )
    parent = _minimax_parent(
        turn=Color.WHITE,
        children={"loss": loss_child, "draw": draw_child},
        all_generated=True,
    )

    parent.backup_from_children(
        branches_with_updated_value={"loss", "draw"},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value is not None
    assert parent.minmax_value.certainty is Certainty.FORCED
    assert parent.minmax_value.over_event is not None
    assert parent.minmax_value.over_event.is_draw()
    assert not parent.is_terminal()


def test_minimax_partial_information_keeps_parent_estimate() -> None:
    loss_child = _minimax_leaf(
        1,
        Value(
            score=-1.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(winner=Color.BLACK),
        ),
    )
    live_child = _minimax_leaf(
        2,
        Value(score=0.2, certainty=Certainty.ESTIMATE),
    )
    parent = _minimax_parent(
        turn=Color.WHITE,
        children={"loss": loss_child, "live": live_child},
        all_generated=False,
    )

    parent.backup_from_children(
        branches_with_updated_value={"loss", "live"},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value is not None
    assert parent.minmax_value.certainty is Certainty.ESTIMATE
    assert parent.minmax_value.over_event is None
    assert not parent.has_exact_value()
    assert not parent.is_terminal()
    assert parent.frontier_branches_in_order() == ["live"]


def test_minimax_partial_draw_child_keeps_parent_estimate_and_live_branch_open() -> (
    None
):
    draw_child = _minimax_leaf(
        1,
        Value(
            score=0.0,
            certainty=Certainty.TERMINAL,
            over_event=_FakeOverEvent(draw=True),
        ),
    )
    live_child = _minimax_leaf(
        2,
        Value(score=0.4, certainty=Certainty.ESTIMATE),
    )
    parent = _minimax_parent(
        turn=Color.WHITE,
        children={"draw": draw_child, "live": live_child},
        all_generated=False,
    )

    parent.backup_from_children(
        branches_with_updated_value={"draw", "live"},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.minmax_value is not None
    assert parent.minmax_value.certainty is Certainty.ESTIMATE
    assert parent.minmax_value.over_event is None
    assert not parent.has_exact_value()
    assert not parent.is_terminal()
    assert parent.frontier_branches_in_order() == ["live"]
