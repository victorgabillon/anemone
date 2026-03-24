"""Principal-variation helpers extracted from ``NodeTreeEvaluationState``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.node_evaluation.common.principal_variation import (
        PrincipalVariationState,
    )


class ChildPrincipalVariationAccess(Protocol):
    """Minimal child evaluation surface needed for PV tracking."""

    @property
    def best_branch_sequence(self) -> list[BranchKey]:
        """Return the child's PV sequence."""
        ...


class ChildTreeEvaluationLookup(Protocol):
    """Resolve child evaluations for PV assembly."""

    def child_tree_evaluation(
        self,
        branch_key: BranchKey,
    ) -> ChildPrincipalVariationAccess | None:
        """Return the child evaluation associated with one branch."""
        ...


def child_pv_version(child: Any) -> int:
    """Return one child's PV version with a conservative fallback."""
    return int(getattr(child.tree_evaluation, "pv_version", 0))


def child_best_branch_sequence(child: Any) -> list[BranchKey]:
    """Return one child's current PV sequence."""
    return list(child.tree_evaluation.best_branch_sequence)


def pv_child_version_for_sequence(
    *,
    sequence: list[BranchKey],
    child_tree_evaluation_getter: ChildTreeEvaluationLookup,
) -> int | None:
    """Return the cached best-child PV version for one PV sequence head."""
    if not sequence:
        return None

    best_child_tree_evaluation = child_tree_evaluation_getter.child_tree_evaluation(
        sequence[0]
    )
    if best_child_tree_evaluation is None:
        return None

    return int(getattr(best_child_tree_evaluation, "pv_version", 0))


def update_best_branch_sequence(
    *,
    pv_state: PrincipalVariationState,
    best_branch_key: BranchKey | None,
    best_child: Any | None,
    branches_with_updated_best_branch_seq: set[BranchKey],
) -> bool:
    """Update the current PV from the best child when needed."""
    return pv_state.try_update_from_best_child(
        best_branch_key=best_branch_key,
        best_child=best_child,
        branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        child_pv_version_getter=child_pv_version,
        child_best_branch_sequence_getter=child_best_branch_sequence,
    )


def set_best_branch_sequence(
    *,
    pv_state: PrincipalVariationState,
    new_sequence: list[BranchKey],
    current_best_child_version: int | None,
) -> bool:
    """Replace PV content and update child-version tracking when it changes."""
    return pv_state.set_sequence(
        new_sequence,
        current_best_child_version=current_best_child_version,
    )


def clear_best_branch_sequence(*, pv_state: PrincipalVariationState) -> bool:
    """Clear stored principal-variation content."""
    return pv_state.clear()
