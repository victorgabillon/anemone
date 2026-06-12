"""Shared vocabulary for direct, backed-up, and candidate node values."""

from typing import Protocol

from valanga.evaluations import Value

from anemone._valanga_types import AnyOverEvent
from anemone.node_evaluation.common.value_candidate import (
    ValueCandidate,
    ValueCandidateSource,
)


class NodeValueEvaluation(Protocol):
    """Minimal value-facing protocol shared by all node-evaluation styles.

    The key terms are:

    * ``direct_value``: immediate evaluator output for this node
    * ``tree_value``: child/subtree-derived value propagated from children
    * ``backed_up_value``: current storage property for ``tree_value``
    * ``get_effective_value_candidate()``: search-facing value plus source,
      accounting for partial versus fully opened nodes
    * ``get_value_candidate()``: value-only view of ``effective_value``
    * ``get_value()``: required canonical value for consumers that need a
      concrete ``Value``
    """

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
    def tree_value(self) -> Value | None:
        """Return the child/subtree-derived value for this node."""
        ...

    @tree_value.setter
    def tree_value(self, value: Value | None) -> None:
        """Set the child/subtree-derived value for this node."""
        ...

    @property
    def effective_value(self) -> Value | None:
        """Return the current search-facing effective value."""
        ...

    @property
    def effective_value_source(self) -> ValueCandidateSource:
        """Return the source for the current effective value."""
        ...

    @property
    def over_event(self) -> AnyOverEvent | None:
        """Return exact outcome metadata when present."""
        ...

    def get_tree_value_candidate(self) -> ValueCandidate:
        """Return the child/subtree-derived value candidate only."""
        ...

    def get_effective_value_candidate(self) -> ValueCandidate:
        """Return the effective candidate and provenance."""
        ...

    def get_value_candidate(self) -> Value | None:
        """Return the best currently available value, or ``None`` if absent."""
        ...

    def get_value(self) -> Value:
        """Return the required canonical ``Value`` for downstream consumers."""
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
