"""Canonical value-access helpers shared by tree-evaluation families."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from anemone.node_evaluation.common import canonical_value

if TYPE_CHECKING:
    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone._valanga_types import AnyOverEvent


class CanonicalNodeValueAccess(Protocol):
    """Minimal node surface needed to expose canonical value helpers."""

    @property
    def direct_value(self) -> Value | None:
        """Return the direct value currently attached to the node."""
        ...

    @property
    def backed_up_value(self) -> Value | None:
        """Return the backed-up value currently attached to the node."""
        ...


class ChildValueCandidate(Protocol):
    """Minimal child evaluation surface needed for value lookup."""

    def get_value_candidate(self) -> Value | None:
        """Return the best available child value candidate."""
        ...


class ChildTreeEvaluationLookup(Protocol):
    """Minimal parent surface needed to resolve child evaluations."""

    def child_tree_evaluation(
        self,
        branch_key: BranchKey,
    ) -> ChildValueCandidate | None:
        """Return the child evaluation associated with one branch."""
        ...


def get_value_candidate(node_eval: CanonicalNodeValueAccess) -> Value | None:
    """Return backed-up value when available, else direct value."""
    return canonical_value.get_value_candidate(
        backed_up_value=node_eval.backed_up_value,
        direct_value=node_eval.direct_value,
    )


def get_value(node_eval: CanonicalNodeValueAccess) -> Value:
    """Return the canonical value for one node."""
    return canonical_value.get_value(
        backed_up_value=node_eval.backed_up_value,
        direct_value=node_eval.direct_value,
    )


def get_score(node_eval: CanonicalNodeValueAccess) -> float:
    """Return the canonical scalar score for one node."""
    return canonical_value.get_score(
        backed_up_value=node_eval.backed_up_value,
        direct_value=node_eval.direct_value,
    )


def has_exact_value(node_eval: CanonicalNodeValueAccess) -> bool:
    """Return whether the canonical candidate value is exact."""
    return canonical_value.is_exact_value(get_value_candidate(node_eval))


def is_terminal(node_eval: CanonicalNodeValueAccess) -> bool:
    """Return whether the canonical candidate is terminal for this node."""
    return canonical_value.is_terminal_value(get_value_candidate(node_eval))


def has_over_event(node_eval: CanonicalNodeValueAccess) -> bool:
    """Return whether the canonical candidate carries exact outcome metadata."""
    return canonical_value.has_over_event(get_value_candidate(node_eval))


def get_over_event_candidate(
    node_eval: CanonicalNodeValueAccess,
) -> AnyOverEvent | None:
    """Return exact outcome metadata from the canonical candidate when present."""
    return canonical_value.get_over_event_candidate(get_value_candidate(node_eval))


def child_value_candidate(
    node_eval: ChildTreeEvaluationLookup,
    branch_key: BranchKey,
) -> Value | None:
    """Return the best available value candidate for one child branch."""
    child_tree_evaluation = node_eval.child_tree_evaluation(branch_key)
    if child_tree_evaluation is None:
        return None
    return child_tree_evaluation.get_value_candidate()
