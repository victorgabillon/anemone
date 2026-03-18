"""Provide reusable principal-variation state for tree-search evaluations."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from valanga import BranchKey


def make_best_branch_sequence_factory() -> list[BranchKey]:
    """Create the default PV sequence container."""
    return []


@dataclass(slots=True)
class PrincipalVariationState:
    """Store the current principal variation and its incremental tracking metadata."""

    best_branch_sequence: list[BranchKey] = field(
        default_factory=make_best_branch_sequence_factory
    )
    pv_version: int = 0
    cached_best_child_version: int | None = None

    def set_sequence(
        self,
        new_sequence: Sequence[BranchKey],
        *,
        current_best_child_version: int | None,
    ) -> bool:
        """Set the PV sequence when its content actually changes."""
        normalized_sequence = list(new_sequence)
        if self.best_branch_sequence == normalized_sequence:
            return False

        self.best_branch_sequence = normalized_sequence
        self.pv_version += 1
        self.cached_best_child_version = (
            current_best_child_version if self.best_branch_sequence else None
        )
        return True

    def clear(self) -> bool:
        """Clear the PV sequence if it is currently non-empty."""
        if not self.best_branch_sequence:
            return False
        return self.set_sequence([], current_best_child_version=None)

    def try_update_from_best_child[ChildT](
        self,
        *,
        best_branch_key: BranchKey | None,
        best_child: ChildT | None,
        branches_with_updated_best_branch_seq: set[BranchKey],
        child_pv_version_getter: Callable[[ChildT], int],
        child_best_branch_sequence_getter: Callable[[ChildT], list[BranchKey]],
    ) -> bool:
        """Refresh PV incrementally from the current best child when needed.

        The caller provides the policy signal via
        ``branches_with_updated_best_branch_seq``. If the stored PV already points
        at a different head, this helper deliberately stays conservative and
        leaves full-head rewrites to the caller's broader backup logic.
        """
        if best_branch_key is None:
            return self.clear()

        if best_branch_key not in branches_with_updated_best_branch_seq:
            return False

        if best_child is None:
            return False

        if self.best_branch_sequence:
            if self.best_branch_sequence[0] != best_branch_key:
                return False

            best_child_version = child_pv_version_getter(best_child)
            if self.cached_best_child_version == best_child_version:
                return False
        else:
            best_child_version = child_pv_version_getter(best_child)

        return self.set_sequence(
            [best_branch_key, *child_best_branch_sequence_getter(best_child)],
            current_best_child_version=best_child_version,
        )
