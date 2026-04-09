"""Adversarial zero-sum objective built on top of ``EvaluationOrdering``."""

from dataclasses import dataclass

from valanga import Color
from valanga.evaluations import Value

from anemone._valanga_types import AnyOverEvent, AnyTurnState
from anemone.values import DEFAULT_EVALUATION_ORDERING, EvaluationOrdering


def _state_turn_must_be_color_error(turn: object) -> TypeError:
    return TypeError(
        "AdversarialZeroSumObjective requires state.turn to be a valanga.Color, "
        f"got {type(turn).__name__}"
    )


def _require_color_turn(state: AnyTurnState) -> Color:
    turn = state.turn
    if not isinstance(turn, Color):
        raise _state_turn_must_be_color_error(turn)
    return turn


@dataclass(frozen=True, slots=True)
class AdversarialZeroSumObjective[StateT: AnyTurnState = AnyTurnState]:
    """State-aware objective adapter for adversarial turn-based semantics.

    This objective delegates reusable ``Value`` comparison/projection rules to
    ``EvaluationOrdering`` and supplies the current node's turn as context.
    """

    evaluation_ordering: EvaluationOrdering = DEFAULT_EVALUATION_ORDERING

    def evaluate_value(self, value: Value, state: StateT) -> float:
        """Project one child ``Value`` using the current node-local perspective."""
        return self.evaluation_ordering.search_sort_key(
            value,
            side_to_move=_require_color_turn(state),
        )

    def semantic_compare(self, left: Value, right: Value, state: StateT) -> int:
        """Compare child ``Value`` objects using current node-local adversarial semantics."""
        return self.evaluation_ordering.semantic_compare(
            left,
            right,
            side_to_move=_require_color_turn(state),
        )

    def terminal_score(self, over_event: AnyOverEvent, state: StateT) -> float:
        """Project terminal outcomes using the current node's side-to-move context."""
        return self.evaluation_ordering.terminal_score(
            over_event,
            perspective=_require_color_turn(state),
        )
