"""Adapters for bridging float evaluators and Value evaluators."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from valanga import State

from anemone.values import Certainty, Value

from .protocols import (
    MasterStateEvaluator,
    OverEventDetector,
)


@runtime_checkable
class _SupportsEvaluate(Protocol):
    """Evaluator exposing a generic per-state scalar evaluation."""

    over: OverEventDetector

    def evaluate(self, state: State) -> float:
        """Return a scalar for one state."""
        ...


@runtime_checkable
class _SupportsEvaluateBatchItems(Protocol):
    """Evaluator exposing batched generic scores."""

    def evaluate_batch_items(self, items: Sequence[Any]) -> list[float]:
        """Return scalar evaluations for a batch of items."""
        ...


@runtime_checkable
class _SupportsValueWhiteBatchItems(Protocol):
    """Evaluator exposing batched white-relative scores."""

    def value_white_batch_items(self, items: Sequence[Any]) -> list[float]:
        """Return white-relative scalars for a batch of items."""
        ...


@dataclass(slots=True)
class FloatToValueEvaluator:
    """Wrap a float-returning evaluator to return Value objects."""

    inner: MasterStateEvaluator | _SupportsEvaluate
    _over: OverEventDetector | None = None

    def __post_init__(self) -> None:
        """Default over detector from the wrapped evaluator when omitted."""
        if self._over is None:
            self._over = self.inner.over

    @property
    def over(self) -> OverEventDetector:
        """Over-event detector associated with this evaluator."""
        assert self._over is not None
        return self._over

    def evaluate(self, state: State) -> Value:
        """Evaluate one state into a Value estimate."""
        if isinstance(self.inner, _SupportsEvaluate):
            score = float(self.inner.evaluate(state))
        else:
            score = float(self.inner.value_white(state))
        return Value(score=score, certainty=Certainty.ESTIMATE, over_event=None)

    def evaluate_batch_items(self, items: Sequence[Any]) -> list[Value]:
        """Evaluate a sequence of items/nodes into Value estimates."""
        if isinstance(self.inner, _SupportsEvaluateBatchItems):
            scores = self.inner.evaluate_batch_items(items)
        elif isinstance(self.inner, _SupportsValueWhiteBatchItems):
            scores = self.inner.value_white_batch_items(items)
        else:
            return [self.evaluate(getattr(item, "state", item)) for item in items]

        return [
            Value(score=float(score), certainty=Certainty.ESTIMATE, over_event=None)
            for score in scores
        ]
