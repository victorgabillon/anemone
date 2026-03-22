"""Adversarial zero-sum Objective built on top of EvaluationOrdering."""

from dataclasses import dataclass

from valanga.evaluations import Value

from anemone._valanga_types import AnyOverEvent, AnyTurnState
from anemone.values import DEFAULT_EVALUATION_ORDERING, EvaluationOrdering


@dataclass(frozen=True, slots=True)
class AdversarialZeroSumObjective[StateT: AnyTurnState = AnyTurnState]:
    """Thin objective adapter for the current turn-based adversarial semantics."""

    evaluation_ordering: EvaluationOrdering = DEFAULT_EVALUATION_ORDERING

    def evaluate_value(self, value: Value, state: StateT) -> float:
        """Return the existing projection-based search ordering key for this node."""
        return self.evaluation_ordering.search_sort_key(
            value,
            side_to_move=state.turn,
        )

    def semantic_compare(self, left: Value, right: Value, state: StateT) -> int:
        """Compare two values using the current side-to-move adversarial semantics."""
        return self.evaluation_ordering.semantic_compare(
            left,
            right,
            side_to_move=state.turn,
        )

    def terminal_score(self, over_event: AnyOverEvent, state: StateT) -> float:
        """Project terminal outcomes using the current side-to-move perspective."""
        return self.evaluation_ordering.terminal_score(
            over_event,
            perspective=state.turn,
        )
