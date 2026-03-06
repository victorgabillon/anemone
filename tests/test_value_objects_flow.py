"""Focused tests for Value-object flow during direct evaluation and backup."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from valanga import Color, OverEvent
from valanga.over_event import HowOver

from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    EvaluationQueries,
    NodeDirectEvaluator,
)
from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation import (
    NodeTreeEvaluation,
)
from anemone.values import Certainty, Value


class _OverDetector:
    def check_obvious_over_events(
        self, state: SimpleNamespace
    ) -> tuple[Any | None, float | None]:
        if state.is_terminal:
            over = OverEvent()
            over.becomes_over(
                how_over=HowOver.WIN,
                who_is_winner=Color.WHITE,
                termination="terminal",
            )
            return (
                over,
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


def _make_node(
    *, node_id: int, turn: Color, base_score: float, is_terminal: bool
) -> Any:
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


def test_terminal_query_routes_to_over_nodes_immediately() -> None:
    """Terminal detection routes node to over_nodes and skips not_over queue."""
    evaluator = NodeDirectEvaluator(master_state_evaluator=_BatchValueEvaluator())
    node = _make_node(node_id=3, turn=Color.WHITE, base_score=0.0, is_terminal=True)
    queries = EvaluationQueries()

    evaluator.add_evaluation_query(node, queries)

    assert queries.over_nodes == [node]
    assert queries.not_over_nodes == []
    assert node.tree_evaluation.direct_value is not None
    assert node.tree_evaluation.direct_value.certainty is Certainty.TERMINAL


def test_add_query_does_not_require_algorithm_node_is_over() -> None:
    """Queue routing uses Value-terminal candidate state, not AlgorithmNode.is_over()."""
    evaluator = NodeDirectEvaluator(master_state_evaluator=_BatchValueEvaluator())
    node = _make_node(node_id=4, turn=Color.WHITE, base_score=0.0, is_terminal=True)

    def _legacy_is_over_should_not_be_called() -> bool:
        raise AssertionError("add_evaluation_query should use is_terminal_candidate()")

    node.is_over = _legacy_is_over_should_not_be_called
    queries = EvaluationQueries()

    evaluator.add_evaluation_query(node, queries)

    assert queries.over_nodes == [node]
    assert queries.not_over_nodes == []


def test_algorithm_node_is_over_uses_terminal_candidate(monkeypatch) -> None:
    """AlgorithmNode.is_over is a compatibility wrapper over terminal candidate semantics."""
    evaluator = NodeDirectEvaluator(master_state_evaluator=_BatchValueEvaluator())
    node = _make_node(node_id=5, turn=Color.WHITE, base_score=0.0, is_terminal=True)

    evaluator.check_obvious_over_events(node)
    assert node.tree_evaluation.is_terminal_candidate()

    def _legacy_tree_eval_is_over(self) -> bool:
        raise AssertionError(
            "AlgorithmNode.is_over should not call tree_evaluation.is_over()"
        )

    monkeypatch.setattr(
        type(node.tree_evaluation),
        "is_over",
        _legacy_tree_eval_is_over,
    )

    assert node.is_over()


def test_minmax_value_is_populated_after_child_backup_and_bridge_holds() -> None:
    """Backup populates minmax Value and keeps float-score bridge aligned."""
    parent = _make_node(node_id=10, turn=Color.WHITE, base_score=0.1, is_terminal=False)
    child = _make_node(node_id=11, turn=Color.BLACK, base_score=0.8, is_terminal=False)

    parent.tree_node.branches_children = {0: child}
    parent.tree_node.all_branches_generated = True

    child.tree_evaluation.direct_value = Value(score=0.8, certainty=Certainty.ESTIMATE)

    parent.tree_evaluation.direct_value = Value(score=0.1, certainty=Certainty.ESTIMATE)
    parent.tree_evaluation.backup_from_children(
        branches_with_updated_value={0},
        branches_with_updated_best_branch_seq=set(),
    )

    assert parent.tree_evaluation.minmax_value is not None
    assert (
        parent.tree_evaluation.get_score() == parent.tree_evaluation.minmax_value.score
    )


def test_get_value_prefers_minmax_else_direct() -> None:
    """Canonical Value getter returns minmax when present, else direct."""
    node = _make_node(node_id=20, turn=Color.WHITE, base_score=0.2, is_terminal=False)

    direct = Value(score=0.2, certainty=Certainty.ESTIMATE)
    node.tree_evaluation.direct_value = direct

    assert node.tree_evaluation.get_value() == direct

    minmax = Value(score=0.6, certainty=Certainty.FORCED)
    node.tree_evaluation.minmax_value = minmax

    assert node.tree_evaluation.get_value() == minmax


def _protocol_score(eval_like: NodeTreeEvaluation[Any]) -> float:
    return eval_like.get_score()


def test_node_tree_evaluation_protocol_exposes_value_api() -> None:
    node = _make_node(node_id=99, turn=Color.WHITE, base_score=0.4, is_terminal=False)
    node.tree_evaluation.direct_value = Value(score=0.4, certainty=Certainty.ESTIMATE)

    assert _protocol_score(node.tree_evaluation) == 0.4
    assert node.tree_evaluation.get_value_candidate() is not None
    assert node.tree_evaluation.get_value().score == 0.4
