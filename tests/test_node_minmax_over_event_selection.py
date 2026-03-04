"""Tests around over-event selection in NodeMinmaxEvaluation."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from valanga import Color

from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.values import Certainty, Value


@dataclass
class FakeOverEvent:
    """Small over-event stub for unit tests."""

    how_over: str | None = None
    who_is_winner: Color | None = None
    termination: str | None = None
    _is_over: bool = False

    def is_over(self) -> bool:
        return self._is_over

    def is_win(self) -> bool:
        return self._is_over and self.who_is_winner is not None

    def is_draw(self) -> bool:
        return self._is_over and self.who_is_winner is None

    def is_winner(self, player: Color) -> bool:
        return self._is_over and self.who_is_winner is player

    def becomes_over(
        self,
        *,
        how_over: str | None,
        who_is_winner: Color | None,
        termination: str | None,
    ) -> None:
        self._is_over = True
        self.how_over = how_over
        self.who_is_winner = who_is_winner
        self.termination = termination


@dataclass
class FakeChildEvaluation:
    """Child evaluation used by the parent NodeMinmaxEvaluation under test."""

    value_white: float
    over_event: FakeOverEvent
    value_white_minmax: float | None = None
    direct_value: Value | None = None
    minmax_value: Value | None = None
    best_branch_sequence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        canonical_value = Value(
            score=self.value_white,
            certainty=(
                Certainty.FORCED if self.over_event.is_over() else Certainty.ESTIMATE
            ),
            over_event=self.over_event if self.over_event.is_over() else None,
        )
        if self.direct_value is None:
            self.direct_value = canonical_value
        if self.minmax_value is None:
            self.minmax_value = self.direct_value
        self.value_white_minmax = self.minmax_value.score

    def set_value(self, score: float) -> None:
        """Keep float bridge and canonical Values aligned in test mutations."""
        self.value_white = score
        canonical_value = Value(
            score=score,
            certainty=(
                Certainty.FORCED if self.over_event.is_over() else Certainty.ESTIMATE
            ),
            over_event=self.over_event if self.over_event.is_over() else None,
        )
        self.direct_value = canonical_value
        self.minmax_value = canonical_value
        self.value_white_minmax = canonical_value.score

    def get_value_white(self) -> float:
        if self.minmax_value is not None:
            return self.minmax_value.score
        assert self.direct_value is not None
        return self.direct_value.score

    def is_winner(self, player: Color) -> bool:
        return self.over_event.is_winner(player)

    def is_draw(self) -> bool:
        return self.over_event.is_draw()


@dataclass
class FakeChildNode:
    """Minimal child node API expected by NodeMinmaxEvaluation."""

    node_id: int
    tree_evaluation: FakeChildEvaluation

    @property
    def tree_node(self) -> Any:
        return SimpleNamespace(id=self.node_id)

    def is_over(self) -> bool:
        return self.tree_evaluation.over_event.is_over()


def _build_parent_eval(
    children: dict[str, FakeChildNode],
) -> NodeMinmaxEvaluation[Any, Any]:
    parent_tree_node = SimpleNamespace(
        id=0,
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children=children,
        all_branches_generated=True,
    )
    evaluation = NodeMinmaxEvaluation(tree_node=parent_tree_node)
    evaluation.over_event = FakeOverEvent()
    return evaluation


def test_becoming_over_prefers_terminal_win_even_if_best_branch_not_over() -> None:
    win_child = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.1,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.WHITE,
                termination="mate",
                _is_over=True,
            ),
        ),
    )
    non_terminal_better_value = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.9,
            over_event=FakeOverEvent(),
        ),
    )

    parent_eval = _build_parent_eval(
        {
            "not_over": non_terminal_better_value,
            "winning_over": win_child,
        }
    )
    # Keep value-ordering with non-terminal branch first.
    parent_eval.branches_sorted_by_value_ = {
        "not_over": (10.0, 0, 2),
        "winning_over": (1.0, 0, 1),
    }

    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_winner(Color.WHITE)
    assert parent_eval.over_event.termination == "mate"


def test_becoming_over_prefers_draw_over_loss_when_all_children_over() -> None:
    draw_child = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.0,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=None,
                termination="stalemate",
                _is_over=True,
            ),
        ),
    )
    loss_child = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=-1.0,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.BLACK,
                termination="mate",
                _is_over=True,
            ),
        ),
    )

    parent_eval = _build_parent_eval({"loss": loss_child, "draw": draw_child})
    parent_eval.branches_sorted_by_value_ = {
        "loss": (5.0, 0, 2),
        "draw": (1.0, 0, 1),
    }

    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_draw()
    assert parent_eval.over_event.termination == "stalemate"


def test_becoming_over_prefers_win_when_multiple_terminal_wins() -> None:
    early_win_child = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.3,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.WHITE,
                termination="mate_in_5",
                _is_over=True,
            ),
        ),
    )
    late_win_child = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=0.2,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.WHITE,
                termination="mate_in_7",
                _is_over=True,
            ),
        ),
    )

    parent_eval = _build_parent_eval(
        {"first_win": early_win_child, "second_win": late_win_child}
    )
    parent_eval.branches_sorted_by_value_ = {
        "first_win": (1.0, 0, 1),
        "second_win": (2.0, 0, 2),
    }

    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_winner(Color.WHITE)
    assert parent_eval.over_event.termination == "mate_in_5"


def test_becoming_over_uses_terminal_loss_when_no_draw_or_win() -> None:
    first_loss = FakeChildNode(
        node_id=1,
        tree_evaluation=FakeChildEvaluation(
            value_white=-0.8,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.BLACK,
                termination="mate_a",
                _is_over=True,
            ),
        ),
    )
    second_loss = FakeChildNode(
        node_id=2,
        tree_evaluation=FakeChildEvaluation(
            value_white=-0.7,
            over_event=FakeOverEvent(
                how_over="forced",
                who_is_winner=Color.BLACK,
                termination="mate_b",
                _is_over=True,
            ),
        ),
    )

    # Intentionally leave branches_sorted_by_value_ empty so fallback order is used.
    parent_eval = _build_parent_eval({"a_loss": first_loss, "b_loss": second_loss})
    parent_eval.becoming_over_from_children()

    assert parent_eval.over_event.is_over()
    assert parent_eval.over_event.is_winner(Color.BLACK)
    # Fallback should be deterministic by branch key (`a_loss` before `b_loss`).
    assert parent_eval.over_event.termination == "mate_a"
