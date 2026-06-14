"""Runtime budgets for materialized opening expansion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OpeningExpansionBudget:
    """Budget consumed by actual materialized tree-edge openings."""

    remaining_branch_openings: int | None = None

    @classmethod
    def unlimited(cls) -> OpeningExpansionBudget:
        """Return an unlimited expansion budget."""
        return cls()

    @classmethod
    def from_tree_limits(
        cls,
        *,
        tree_branch_limit: int | None,
        current_branch_count: int,
    ) -> OpeningExpansionBudget:
        """Create a runtime budget from current tree-count limits."""
        if tree_branch_limit is None:
            return cls.unlimited()
        remaining_branch_openings = max(tree_branch_limit - current_branch_count, 0)
        return cls(remaining_branch_openings=remaining_branch_openings)

    def can_open_branch(self) -> bool:
        """Return whether one more branch edge may be materialized."""
        return (
            self.remaining_branch_openings is None or self.remaining_branch_openings > 0
        )

    def reserve_branch_opening(self) -> bool:
        """Reserve capacity for one materialized branch edge."""
        if not self.can_open_branch():
            return False
        if self.remaining_branch_openings is not None:
            self.remaining_branch_openings -= 1
        return True


def reserve_branch_opening(
    budget: OpeningExpansionBudget | None,
) -> bool:
    """Reserve one branch opening from ``budget`` when one is provided."""
    if budget is None:
        return True
    return budget.reserve_branch_opening()


__all__ = [
    "OpeningExpansionBudget",
    "reserve_branch_opening",
]
