"""Adversarial zero-sum Objective built on top of EvaluationOrdering."""

from dataclasses import dataclass

from valanga import OverEvent, TurnState

from anemone.values import DEFAULT_EVALUATION_ORDERING, EvaluationOrdering, Value


@dataclass(frozen=True, slots=True)
class AdversarialZeroSumObjective[StateT: TurnState = TurnState]:
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

    def terminal_score(self, over_event: OverEvent, state: StateT) -> float:
        """Project terminal outcomes using the current side-to-move perspective."""
        return self.evaluation_ordering.terminal_score(
            over_event,
            perspective=state.turn,
        )
