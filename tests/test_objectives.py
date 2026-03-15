"""Tests for the Objective abstraction and its minimax integration."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from valanga import Color

from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.objectives import AdversarialZeroSumObjective
from anemone.values import EvaluationOrdering
from valanga.evaluations import Certainty, Value


@dataclass(frozen=True)
class _FakeOverEvent:
    winner: Color | None = None
    draw: bool = False

    def is_over(self) -> bool:
        return self.draw or self.winner is not None

    def is_draw(self) -> bool:
        return self.draw

    def is_winner(self, player: Color) -> bool:
        return self.winner == player


@dataclass
class _ReverseObjective:
    """Test objective that prefers lower scores regardless of side to move."""

    def evaluate_value(self, value: Value, state: Any) -> float:
        del state
        return value.score

    def semantic_compare(self, left: Value, right: Value, state: Any) -> int:
        del state
        if left.score < right.score:
            return 1
        if left.score > right.score:
            return -1
        return 0

    def terminal_score(self, over_event: Any, state: Any) -> float:
        del over_event, state
        return 0.0


def _state(turn: Color) -> Any:
    return SimpleNamespace(turn=turn)


def _leaf(node_id: int, score: float) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=_state(Color.WHITE),
        branches_children={},
        all_branches_generated=True,
    )
    tree_evaluation = NodeMinmaxEvaluation(tree_node=tree_node)
    value = Value(score=score, certainty=Certainty.ESTIMATE)
    tree_evaluation.direct_value = value
    tree_evaluation.minmax_value = value
    return SimpleNamespace(tree_node=tree_node, tree_evaluation=tree_evaluation)


def _exact_leaf(
    node_id: int,
    *,
    turn: Color,
    score: float,
    over_event: _FakeOverEvent,
    pv_length: int,
) -> Any:
    tree_node = SimpleNamespace(
        id=node_id,
        state=_state(turn),
        branches_children={},
        all_branches_generated=True,
    )
    tree_evaluation = NodeMinmaxEvaluation(tree_node=tree_node)
    value = Value(
        score=score,
        certainty=Certainty.FORCED,
        over_event=over_event,
    )
    tree_evaluation.direct_value = value
    tree_evaluation.minmax_value = value
    tree_evaluation.best_branch_sequence = list(range(pv_length))
    return SimpleNamespace(tree_node=tree_node, tree_evaluation=tree_evaluation)


def test_adversarial_objective_matches_existing_ordering_adapter() -> None:
    ordering = EvaluationOrdering(win_score=1.0, draw_score=0.0, loss_score=-1.0)
    objective = AdversarialZeroSumObjective(evaluation_ordering=ordering)
    win = Value(
        score=-99.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    estimate = Value(score=3.0, certainty=Certainty.ESTIMATE)
    state = _state(Color.WHITE)

    assert objective.evaluate_value(win, state) == ordering.search_sort_key(
        win,
        side_to_move=Color.WHITE,
    )
    assert objective.semantic_compare(
        win, estimate, state
    ) == ordering.semantic_compare(
        win,
        estimate,
        side_to_move=Color.WHITE,
    )
    assert objective.terminal_score(
        _FakeOverEvent(winner=Color.WHITE),
        state,
    ) == ordering.terminal_score(
        _FakeOverEvent(winner=Color.WHITE),
        perspective=Color.WHITE,
    )


def test_adversarial_objective_interpretation_depends_on_turn() -> None:
    objective = AdversarialZeroSumObjective()
    lower = Value(score=0.2, certainty=Certainty.ESTIMATE)
    higher = Value(score=0.8, certainty=Certainty.ESTIMATE)

    assert objective.semantic_compare(lower, higher, _state(Color.WHITE)) < 0
    assert objective.semantic_compare(lower, higher, _state(Color.BLACK)) > 0


def test_adversarial_objective_terminal_order_matches_existing_behavior() -> None:
    objective = AdversarialZeroSumObjective()
    win = Value(
        score=-5.0,
        certainty=Certainty.FORCED,
        over_event=_FakeOverEvent(winner=Color.WHITE),
    )
    draw = Value(
        score=99.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(draw=True),
    )
    loss = Value(
        score=5.0,
        certainty=Certainty.TERMINAL,
        over_event=_FakeOverEvent(winner=Color.BLACK),
    )
    state = _state(Color.WHITE)

    assert objective.semantic_compare(win, draw, state) > 0
    assert objective.semantic_compare(draw, loss, state) > 0


def test_minimax_best_branch_uses_injected_objective() -> None:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=_state(Color.WHITE),
        branches_children={0: _leaf(1, 0.9), 1: _leaf(2, 0.1)},
        all_branches_generated=True,
    )
    parent = NodeMinmaxEvaluation(
        tree_node=parent_tree_node,
        objective=_ReverseObjective(),
    )
    parent.direct_value = Value(score=0.0, certainty=Certainty.ESTIMATE)

    parent.backup_from_children(
        branches_with_updated_value={0, 1},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.best_branch() == 1
    assert parent.minmax_value == Value(score=0.1, certainty=Certainty.ESTIMATE)


def test_minimax_branch_order_prefers_shorter_exact_win_lines() -> None:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=_state(Color.WHITE),
        branches_children={
            0: _exact_leaf(
                1,
                turn=Color.BLACK,
                score=1.0,
                over_event=_FakeOverEvent(winner=Color.WHITE),
                pv_length=3,
            ),
            1: _exact_leaf(
                2,
                turn=Color.BLACK,
                score=1.0,
                over_event=_FakeOverEvent(winner=Color.WHITE),
                pv_length=1,
            ),
        },
        all_branches_generated=True,
    )
    parent = NodeMinmaxEvaluation(tree_node=parent_tree_node)

    parent.update_branches_values({0, 1})

    assert parent.best_branch() == 1


def test_minimax_branch_order_prefers_longer_exact_loss_lines() -> None:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=_state(Color.WHITE),
        branches_children={
            0: _exact_leaf(
                1,
                turn=Color.BLACK,
                score=-1.0,
                over_event=_FakeOverEvent(winner=Color.BLACK),
                pv_length=1,
            ),
            1: _exact_leaf(
                2,
                turn=Color.BLACK,
                score=-1.0,
                over_event=_FakeOverEvent(winner=Color.BLACK),
                pv_length=3,
            ),
        },
        all_branches_generated=True,
    )
    parent = NodeMinmaxEvaluation(tree_node=parent_tree_node)

    parent.update_branches_values({0, 1})

    assert parent.best_branch() == 1
