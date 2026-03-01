"""Focused tests for Value-object flow during direct evaluation and backup."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from valanga import Color

from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    EvaluationQueries,
    NodeDirectEvaluator,
)
from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.values import Certainty, Value


class _OverDetector:
    def check_obvious_over_events(
        self, state: SimpleNamespace
    ) -> tuple[Any | None, float | None]:
        if state.is_terminal:
            return (
                SimpleNamespace(
                    how_over="mate",
                    who_is_winner=Color.WHITE,
                    termination="terminal",
                ),
                1.0,
            )
        return None, None


class _MutableOverEvent:
    def __init__(self) -> None:
        self._is_over = False
        self.how_over: Any | None = None
        self.who_is_winner: Any | None = None
        self.termination: Any | None = None

    def becomes_over(self, how_over: Any, who_is_winner: Any, termination: Any) -> None:
        self._is_over = True
        self.how_over = how_over
        self.who_is_winner = who_is_winner
        self.termination = termination

    def is_over(self) -> bool:
        return self._is_over

    def is_winner(self, player: Color) -> bool:
        return self._is_over and self.who_is_winner is player

    def is_draw(self) -> bool:
        return self._is_over and self.who_is_winner is None


class _BatchValueEvaluator:
    over = _OverDetector()

    def evaluate(self, state: Any) -> Value:
        return Value(score=state.base_score, certainty=Certainty.ESTIMATE)

    def evaluate_batch_items(self, items: Sequence[Any]) -> list[Value]:
        return [
            Value(score=node.state.base_score, certainty=Certainty.ESTIMATE)
            for node in items
        ]


def _make_node(*, node_id: int, turn: Color, base_score: float, is_terminal: bool) -> Any:
    state = SimpleNamespace(turn=turn, base_score=base_score, is_terminal=is_terminal)
    tree_node = SimpleNamespace(
        id=node_id,
        state=state,
        branches_children={},
        all_branches_generated=False,
    )
    tree_evaluation = NodeMinmaxEvaluation(tree_node=tree_node)
    tree_evaluation.over_event = _MutableOverEvent()
    return SimpleNamespace(
        state=state,
        tree_node=tree_node,
        tree_evaluation=tree_evaluation,
        tree_depth=0,
        is_over=tree_evaluation.is_over,
    )


def test_non_terminal_evaluation_sets_direct_value() -> None:
    """Non-terminal direct evaluation stores a non-terminal Value object."""
    evaluator = NodeDirectEvaluator(master_state_evaluator=_BatchValueEvaluator())
    node = _make_node(node_id=1, turn=Color.WHITE, base_score=0.3, is_terminal=False)
    queries = EvaluationQueries()

    evaluator.add_evaluation_query(node, queries)
    evaluator.evaluate_all_queried_nodes(queries)

    assert node.tree_evaluation.direct_value is not None
    assert node.tree_evaluation.direct_value.certainty is Certainty.ESTIMATE
    assert node.tree_evaluation.direct_value.over_event is None


def test_terminal_detection_sets_terminal_direct_value() -> None:
    """Terminal detection stores a terminal Value with over-event metadata."""
    evaluator = NodeDirectEvaluator(master_state_evaluator=_BatchValueEvaluator())
    node = _make_node(node_id=2, turn=Color.WHITE, base_score=0.0, is_terminal=True)
    queries = EvaluationQueries()

    evaluator.add_evaluation_query(node, queries)
    evaluator.evaluate_all_queried_nodes(queries)

    assert node.tree_evaluation.direct_value is not None
    assert node.tree_evaluation.direct_value.certainty is Certainty.TERMINAL
    assert node.tree_evaluation.direct_value.over_event is not None


def test_minmax_value_is_populated_after_child_backup_and_bridge_holds() -> None:
    """Backup populates minmax Value and keeps float-score bridge aligned."""
    parent = _make_node(node_id=10, turn=Color.WHITE, base_score=0.1, is_terminal=False)
    child = _make_node(node_id=11, turn=Color.BLACK, base_score=0.8, is_terminal=False)

    parent.tree_node.branches_children = {0: child}
    parent.tree_node.all_branches_generated = True

    child.tree_evaluation.direct_value = Value(score=0.8, certainty=Certainty.ESTIMATE)
    child.tree_evaluation.set_evaluation(0.8)

    parent.tree_evaluation.direct_value = Value(score=0.1, certainty=Certainty.ESTIMATE)
    parent.tree_evaluation.set_evaluation(0.1)
    parent.tree_evaluation.minmax_value_update_from_children(branches_with_updated_value={0})

    assert parent.tree_evaluation.minmax_value is not None
    assert (
        parent.tree_evaluation.value_white_minmax
        == parent.tree_evaluation.minmax_value.score
    )


def test_get_value_prefers_minmax_else_direct() -> None:
    """Canonical Value getter returns minmax when present, else direct."""
    node = _make_node(node_id=20, turn=Color.WHITE, base_score=0.2, is_terminal=False)

    direct = Value(score=0.2, certainty=Certainty.ESTIMATE)
    node.tree_evaluation.direct_value = direct
    node.tree_evaluation.set_evaluation(0.2)

    assert node.tree_evaluation.get_value() == direct

    minmax = Value(score=0.6, certainty=Certainty.FORCED)
    node.tree_evaluation.minmax_value = minmax
    node.tree_evaluation.value_white_minmax = minmax.score

    assert node.tree_evaluation.get_value() == minmax
