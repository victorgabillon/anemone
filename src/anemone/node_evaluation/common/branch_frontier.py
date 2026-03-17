"""Provide a generic branch-frontier capability and state helper."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from valanga import BranchKey


@runtime_checkable
class BranchFrontierAware(Protocol):
    """Evaluation capability exposing search-relevant child branches."""

    def on_branch_opened(self, branch: BranchKey) -> None:
        """Record that a child branch has been opened and is frontier-relevant."""
        ...

    def sync_branch_frontier(self, branches_to_refresh: set[BranchKey]) -> None:
        """Refresh frontier membership after child-value updates."""
        ...

    def has_frontier_branches(self) -> bool:
        """Return whether any child branches remain frontier-relevant."""
        ...

    def frontier_branches_in_order(self) -> list[BranchKey]:
        """Return frontier branches ordered by current child-preference semantics."""
        ...


@dataclass(slots=True)
class BranchFrontierState:
    """Reusable set-based bookkeeping for search-relevant child branches."""

    frontier_branches: set[BranchKey] = field(default_factory=set)

    def on_branch_opened(self, branch: BranchKey) -> None:
        """Add a newly opened child branch to the frontier."""
        self.frontier_branches.add(branch)

    def clear(self) -> None:
        """Remove every branch from the frontier."""
        self.frontier_branches.clear()

    def has_frontier_branches(self) -> bool:
        """Return whether the frontier currently contains any branch."""
        return bool(self.frontier_branches)

    def set_branch_relevance(self, branch: BranchKey, *, relevant: bool) -> None:
        """Update whether one branch belongs to the frontier."""
        if relevant:
            self.frontier_branches.add(branch)
            return
        self.frontier_branches.discard(branch)

    def sync_with_current_state(
        self,
        *,
        branches_to_refresh: Iterable[BranchKey],
        should_remain_in_frontier: Callable[[BranchKey], bool],
    ) -> None:
        """Refresh frontier membership using evaluation-specific relevance rules."""
        for branch in branches_to_refresh:
            self.set_branch_relevance(
                branch,
                relevant=should_remain_in_frontier(branch),
            )

    def ordered_frontier_branches(
        self,
        ordered_candidate_branches: Iterable[BranchKey],
    ) -> list[BranchKey]:
        """Project frontier membership onto a caller-supplied branch ordering."""
        ordered_frontier: list[BranchKey] = []
        seen: set[BranchKey] = set()
        for branch in ordered_candidate_branches:
            if branch in seen:
                continue
            seen.add(branch)
            if branch in self.frontier_branches:
                ordered_frontier.append(branch)
        return ordered_frontier


def require_branch_frontier_aware(node_evaluation: object) -> BranchFrontierAware:
    """Return a branch-frontier-aware evaluation or raise a clear type error."""
    if not isinstance(node_evaluation, BranchFrontierAware):
        raise TypeError(
            f"Expected a BranchFrontierAware evaluation, got {type(node_evaluation)!r}."
        )
    return node_evaluation
