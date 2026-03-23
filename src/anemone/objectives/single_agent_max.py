"""Single-agent maximizing objective implementation."""

from dataclasses import dataclass

from valanga import State
from valanga.evaluations import Certainty, Value

from anemone._valanga_types import AnyOverEvent


@dataclass(frozen=True, slots=True)
class SingleAgentMaxObjective[StateT: State = State]:
    """State-aware objective for simple single-agent max semantics.

    This objective defines node-local comparison/projection rules directly rather
    than delegating to ``EvaluationOrdering``. It still operates at the ``Value``
    semantics layer, not the branch-cache layer.
    """

    terminal_score_value: float = 0.0

    def evaluate_value(self, value: Value, state: StateT) -> float:
        """Project one ``Value`` to the scalar used for node-local max decisions."""
        del state
        return value.score

    def semantic_compare(self, left: Value, right: Value, state: StateT) -> int:
        """Compare child ``Value`` objects for this node's max objective semantics."""
        del state
        if left.score > right.score:
            return 1
        if left.score < right.score:
            return -1

        left_exact = self._is_exact_for_tie_break(left)
        right_exact = self._is_exact_for_tie_break(right)
        if left_exact and not right_exact:
            return 1
        if right_exact and not left_exact:
            return -1
        return 0

    def terminal_score(self, over_event: AnyOverEvent, state: StateT) -> float:
        """Return the terminal scalar score convention for this objective."""
        del over_event, state
        return self.terminal_score_value

    def _is_exact_for_tie_break(self, value: Value) -> bool:
        """Treat FORCED and TERMINAL equally as exact values for tie-breaking.

        This does not say anything about whether the node's own state is
        terminal; it is only about exactness in the objective ordering.
        """
        return value.certainty in {Certainty.TERMINAL, Certainty.FORCED}
