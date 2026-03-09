"""Single-agent maximizing Objective implementation."""

from dataclasses import dataclass

from valanga import OverEvent, State
from valanga.evaluations import Certainty, Value


@dataclass(frozen=True, slots=True)
class SingleAgentMaxObjective[StateT: State = State]:
    """Interpret canonical Value objects with simple single-agent max semantics."""

    terminal_score_value: float = 0.0

    def evaluate_value(self, value: Value, state: StateT) -> float:
        """Project a Value to the scalar used for single-agent max decisions."""
        del state
        return value.score

    def semantic_compare(self, left: Value, right: Value, state: StateT) -> int:
        """Prefer larger scores, with terminal-like values winning exact-score ties."""
        del state
        if left.score > right.score:
            return 1
        if left.score < right.score:
            return -1

        left_terminal = self._is_terminal_like(left)
        right_terminal = self._is_terminal_like(right)
        if left_terminal and not right_terminal:
            return 1
        if right_terminal and not left_terminal:
            return -1
        return 0

    def terminal_score(self, over_event: OverEvent, state: StateT) -> float:
        """Return the explicit score convention for terminal single-agent states."""
        del over_event, state
        return self.terminal_score_value

    def _is_terminal_like(self, value: Value) -> bool:
        return value.certainty in {Certainty.TERMINAL, Certainty.FORCED}
