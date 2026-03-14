"""Provide a minimal game-agnostic node value protocol."""

from typing import Protocol

from valanga import OverEvent
from valanga.evaluations import Value


class NodeValueEvaluation(Protocol):
    """Minimal canonical-Value surface shared by all node-evaluation styles."""

    @property
    def direct_value(self) -> Value | None:
        """Return the canonical state-evaluator value for this node."""
        ...

    @direct_value.setter
    def direct_value(self, value: Value | None) -> None:
        """Set the canonical state-evaluator value for this node."""
        ...

    @property
    def backed_up_value(self) -> Value | None:
        """Return the canonical subtree-backed-up value for this node."""
        ...

    @backed_up_value.setter
    def backed_up_value(self, value: Value | None) -> None:
        """Set the canonical subtree-backed-up value for this node."""
        ...

    @property
    def over_event(self) -> OverEvent | None:
        """Return exact outcome metadata when present."""
        ...

    def get_value_candidate(self) -> Value | None:
        """Return backed-up value when available, else direct value."""
        ...

    def get_value(self) -> Value:
        """Return canonical Value used by downstream consumers."""
        ...

    def get_score(self) -> float:
        """Return canonical scalar score from the canonical Value."""
        ...

    def has_exact_value(self) -> bool:
        """Return whether the candidate Value is exact."""
        ...

    def is_terminal(self) -> bool:
        """Return whether the candidate Value says this node's own state is terminal."""
        ...

    def has_over_event(self) -> bool:
        """Return whether the candidate Value carries exact outcome metadata."""
        ...
