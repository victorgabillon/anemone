"""Canonical value-access helpers shared by tree-evaluation families."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from anemone.node_evaluation.common import canonical_value

if TYPE_CHECKING:
    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone._valanga_types import AnyOverEvent
    from anemone.node_evaluation.common.value_candidate import ValueCandidate


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

    @property
    def tree_value(self) -> Value | None:
        """Return the child/subtree-derived value currently attached to the node."""
        ...

    @property
    def all_branches_generated(self) -> bool:
        """Return whether this node has generated all legal child branches."""
        ...

    def compare_candidate_values(self, left: Value, right: Value) -> int:
        """Compare two values using node-local objective semantics."""
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
    """Return effective value when present for current opening completeness."""
    return get_effective_value_candidate(node_eval).value


def get_tree_value_candidate(node_eval: CanonicalNodeValueAccess) -> ValueCandidate:
    """Return the child/subtree-derived value candidate only."""
    return canonical_value.get_tree_value_candidate(tree_value=node_eval.tree_value)


def get_effective_value_candidate(
    node_eval: CanonicalNodeValueAccess,
) -> ValueCandidate:
    """Return effective value and source for current opening completeness."""
    return canonical_value.get_effective_value_candidate(
        tree_value=node_eval.tree_value,
        direct_value=node_eval.direct_value,
        all_branches_generated=node_eval.all_branches_generated,
        semantic_compare=node_eval.compare_candidate_values,
    )


def get_value(node_eval: CanonicalNodeValueAccess) -> Value:
    """Return the canonical value for one node."""
    return canonical_value.require_value(get_value_candidate(node_eval))


def get_score(node_eval: CanonicalNodeValueAccess) -> float:
    """Return the canonical scalar score for one node."""
    return get_value(node_eval).score


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
