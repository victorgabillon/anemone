"""Shared test doubles for tree-evaluation-focused unit tests."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from valanga import Color

from valanga.evaluations import Certainty, Value


def set_estimate_value(
    node_eval: "Any",
    *,
    score: float,
    also_minmax: bool = True,
) -> None:
    """Set explicit ESTIMATE Value on a node evaluation test fixture."""
    value = Value(
        score=score,
        certainty=Certainty.ESTIMATE,
        over_event=None,
    )
    node_eval.direct_value = value
    if also_minmax:
        node_eval.minmax_value = value


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
    """Child evaluation used by NodeMinmaxEvaluation unit tests."""

    value_white: float
    over_event: FakeOverEvent = field(default_factory=FakeOverEvent)
    direct_value: Value | None = None
    minmax_value: Value | None = None
    best_branch_sequence: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        canonical_value = Value(
            score=self.value_white,
            certainty=(
                Certainty.TERMINAL if self.over_event.is_over() else Certainty.ESTIMATE
            ),
            over_event=self.over_event if self.over_event.is_over() else None,
        )
        if self.direct_value is None:
            self.direct_value = canonical_value
        if self.minmax_value is None:
            self.minmax_value = self.direct_value

    def set_value(self, score: float) -> None:
        """Keep float bridge and canonical Values aligned in test mutations."""
        self.value_white = score
        canonical_value = Value(
            score=score,
            certainty=(
                Certainty.TERMINAL if self.over_event.is_over() else Certainty.ESTIMATE
            ),
            over_event=self.over_event if self.over_event.is_over() else None,
        )
        self.direct_value = canonical_value
        self.minmax_value = canonical_value

    def get_value_candidate(self) -> Value | None:
        if self.minmax_value is not None:
            return self.minmax_value
        return self.direct_value

    def get_value(self) -> Value:
        candidate = self.get_value_candidate()
        assert candidate is not None, "FakeChildEvaluation has no Value candidate"
        return candidate

    def get_score(self) -> float:
        return self.get_value().score

    def has_exact_value(self) -> bool:
        value = self.get_value_candidate()
        return value is not None and value.certainty in (
            Certainty.TERMINAL,
            Certainty.FORCED,
        )

    def is_terminal(self) -> bool:
        value = self.get_value_candidate()
        return value is not None and value.certainty == Certainty.TERMINAL

    def has_over_event(self) -> bool:
        value = self.get_value_candidate()
        return value is not None and value.over_event is not None

    def _effective_over_event(self) -> FakeOverEvent | None:
        candidate = self.get_value_candidate()
        if candidate is not None and candidate.over_event is not None:
            return candidate.over_event
        return self.over_event

    def is_winner(self, player: Color) -> bool:
        over = self._effective_over_event()
        return over is not None and over.is_winner(player)

    def is_draw(self) -> bool:
        over = self._effective_over_event()
        return over is not None and over.is_draw()


@dataclass
class FakeChildNode:
    """Minimal child node API expected by NodeMinmaxEvaluation."""

    node_id: int
    tree_evaluation: FakeChildEvaluation

    @property
    def tree_node(self) -> Any:
        return SimpleNamespace(id=self.node_id)

    def is_over(self) -> bool:
        return self.tree_evaluation.is_terminal()
