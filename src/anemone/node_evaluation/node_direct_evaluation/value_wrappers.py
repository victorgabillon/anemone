"""Adapters for bridging float evaluators and Value evaluators."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from valanga import OverEvent, State

from anemone.values import Certainty, Value


class OverEventDetector(Protocol):
    """Protocol for detecting over events in a game state."""

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        """Return an over event and evaluation if the state is terminal."""
        ...


@dataclass(slots=True)
class FloatToValueEvaluator:
    """Wrap a float-returning evaluator to return Value objects."""

    inner: object
    over: OverEventDetector | None = None

    def __post_init__(self) -> None:
        """Default the over detector from the wrapped evaluator when available."""
        if self.over is None and hasattr(self.inner, "over"):
            self.over = getattr(self.inner, "over")
        assert self.over is not None

    def evaluate(self, state: State) -> Value:
        """Evaluate one state into a Value estimate."""
        if hasattr(self.inner, "evaluate"):
            score_raw = getattr(self.inner, "evaluate")(state)
        else:
            score_raw = getattr(self.inner, "value_white")(state)
        score = float(score_raw)
        return Value(score=score, certainty=Certainty.ESTIMATE, over_event=None)

    def evaluate_batch_items(self, items: Sequence[Any]) -> list[Value]:
        """Evaluate a sequence of items/nodes into Value estimates."""
        if hasattr(self.inner, "evaluate_batch_items"):
            batch_values = getattr(self.inner, "evaluate_batch_items")(items)
            return [
                Value(score=float(score), certainty=Certainty.ESTIMATE, over_event=None)
                for score in batch_values
            ]

        if hasattr(self.inner, "value_white_batch_items"):
            batch_values = getattr(self.inner, "value_white_batch_items")(items)
            return [
                Value(score=float(score), certainty=Certainty.ESTIMATE, over_event=None)
                for score in batch_values
            ]

        values: list[Value] = []
        for item in items:
            state = getattr(item, "state", item)
            values.append(self.evaluate(state))
        return values
