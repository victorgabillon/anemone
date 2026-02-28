"""Shared protocols for direct node evaluation."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from valanga import OverEvent, State
from valanga.evaluations import EvalItem

from anemone.values import Value


class OverEventDetector(Protocol):
    """Protocol for detecting over events in a game state."""

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        """Return an over event and evaluation if the state is terminal."""
        ...


if TYPE_CHECKING:

    class MasterStateEvaluator(Protocol):
        """Protocol for evaluating a state into a white-relative float."""

        @property
        def over(self) -> OverEventDetector:
            """Over-event detector associated with this evaluator."""
            ...

        def value_white(self, state: State) -> float:
            """Evaluate a single state from white's perspective."""
            ...

        def value_white_batch_items[ItemStateT: State](
            self, items: Sequence[EvalItem[ItemStateT]]
        ) -> list[float]:
            """Evaluate a batch of items, defaulting to single-state calls."""
            return [self.value_white(it.state) for it in items]

else:

    class MasterStateEvaluator(Protocol):
        """Protocol for evaluating a state into a white-relative float."""

        over: OverEventDetector

        def value_white(self, state: State) -> float:
            """Evaluate a single state from white's perspective."""
            ...

        def value_white_batch_items[ItemStateT: State](
            self, items: Sequence[EvalItem[ItemStateT]]
        ) -> list[float]:
            """Evaluate a batch of items, defaulting to single-state calls."""
            return [self.value_white(it.state) for it in items]


if TYPE_CHECKING:

    @runtime_checkable
    class MasterStateValueEvaluator(Protocol):
        """Protocol for evaluating a state into a Value object."""

        @property
        def over(self) -> OverEventDetector:
            """Over-event detector associated with this evaluator."""
            ...

        def evaluate(self, state: State) -> Value:
            """Evaluate one state and return a Value object."""
            ...

        def evaluate_batch_items[ItemStateT: State](
            self, items: Sequence[EvalItem[ItemStateT]]
        ) -> list[Value]:
            """Evaluate a batch of items, defaulting to single-state calls."""
            return [self.evaluate(it.state) for it in items]

else:

    @runtime_checkable
    class MasterStateValueEvaluator(Protocol):
        """Protocol for evaluating a state into a Value object."""

        over: OverEventDetector

        def evaluate(self, state: State) -> Value:
            """Evaluate one state and return a Value object."""
            ...

        def evaluate_batch_items[ItemStateT: State](
            self, items: Sequence[EvalItem[ItemStateT]]
        ) -> list[Value]:
            """Evaluate a batch of items, defaulting to single-state calls."""
            return [self.evaluate(it.state) for it in items]
