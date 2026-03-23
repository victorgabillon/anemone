"""Protocols for state-aware semantic interpretation of canonical ``Value`` objects."""

from typing import Protocol

from valanga import State
from valanga.evaluations import Value

from anemone._valanga_types import AnyOverEvent


class Objective[StateT: State = State](Protocol):
    """State-aware adapter for node-local ``Value`` semantics.

    Objectives interpret canonical ``Value`` objects in the context of one node's
    state. They decide how that node compares child values or projects them to a
    scalar, but they do not own branch-order caches.
    """

    def evaluate_value(self, value: Value, state: StateT) -> float:
        """Project a canonical ``Value`` to the scalar used in node-local ordering."""
        ...

    def semantic_compare(self, left: Value, right: Value, state: StateT) -> int:
        """Compare two ``Value`` objects using this node's objective semantics."""
        ...

    def terminal_score(self, over_event: AnyOverEvent, state: StateT) -> float:
        """Project one terminal outcome to this objective's scalar score."""
        ...
