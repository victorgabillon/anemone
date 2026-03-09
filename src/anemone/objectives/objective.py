"""Protocols for semantic interpretation of canonical Value objects."""

from typing import Protocol

from valanga import OverEvent, State
from valanga.evaluations import Value


class Objective[StateT: State = State](Protocol):
    """Interpret canonical Value objects for a node-local search objective family."""

    def evaluate_value(self, value: Value, state: StateT) -> float:
        """Project a canonical Value to the scalar used for node-local ordering."""
        ...

    def semantic_compare(self, left: Value, right: Value, state: StateT) -> int:
        """Compare two Values using the semantics induced by the given state."""
        ...

    def terminal_score(self, over_event: OverEvent, state: StateT) -> float:
        """Project a terminal over-event to the scalar score used by this objective."""
        ...
