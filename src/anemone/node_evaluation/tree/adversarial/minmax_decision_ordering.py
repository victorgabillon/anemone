"""Provide minimax-specific decision-ordering bookkeeping."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cmp_to_key

from valanga import BranchKey
from valanga.evaluations import Value

from anemone.utils.my_value_sorted_dict import sort_dic

type BranchSortValue = tuple[float, int, int]
type BranchSortValueGetter = Callable[[BranchKey], BranchSortValue]
type ChildValueCandidateGetter = Callable[[BranchKey], Value | None]
type SemanticCompare = Callable[[Value, Value], int]


def make_branches_sorted_by_value_factory() -> dict[BranchKey, BranchSortValue]:
    """Create the cached minimax branch-ordering storage."""
    return {}


@dataclass(slots=True)
class MinmaxDecisionOrderingState:
    """Store minimax child ranking state and derive decision ordering from it."""

    branches_sorted_by_value: dict[BranchKey, BranchSortValue] = field(
        default_factory=make_branches_sorted_by_value_factory
    )

    def record_sort_value_of_child(
        self,
        branch_key: BranchKey,
        *,
        branch_sort_value_getter: BranchSortValueGetter,
    ) -> None:
        """Refresh the cached search-order tuple for one child branch."""
        self.branches_sorted_by_value[branch_key] = branch_sort_value_getter(branch_key)

    def update_branches_values(
        self,
        branches_to_consider: Iterable[BranchKey],
        *,
        branch_sort_value_getter: BranchSortValueGetter,
    ) -> None:
        """Refresh cached search-order tuples for the provided branches."""
        for branch_key in branches_to_consider:
            self.record_sort_value_of_child(
                branch_key,
                branch_sort_value_getter=branch_sort_value_getter,
            )
        self.branches_sorted_by_value = sort_dic(self.branches_sorted_by_value)

    def best_branch_value(self) -> BranchSortValue | None:
        """Return the cached search-order tuple of the current best search branch."""
        if not self.branches_sorted_by_value:
            return None
        return next(iter(self.branches_sorted_by_value.values()))

    def decision_ordered_branches(
        self,
        *,
        child_value_candidate_getter: ChildValueCandidateGetter,
        semantic_compare: SemanticCompare,
    ) -> list[BranchKey]:
        """Return child branches ordered by current minimax decision semantics."""
        candidates: list[tuple[BranchKey, Value, BranchSortValue]] = []
        for branch_key, sort_value in self.branches_sorted_by_value.items():
            child_value = child_value_candidate_getter(branch_key)
            if child_value is None:
                continue
            candidates.append((branch_key, child_value, sort_value))

        def _cmp(
            left: tuple[BranchKey, Value, BranchSortValue],
            right: tuple[BranchKey, Value, BranchSortValue],
        ) -> int:
            semantic = semantic_compare(left[1], right[1])
            if semantic != 0:
                return -semantic
            if left[2] < right[2]:
                return -1
            if left[2] > right[2]:
                return 1
            return (
                -1
                if str(left[0]) < str(right[0])
                else (1 if str(left[0]) > str(right[0]) else 0)
            )

        return [
            branch_key for branch_key, _, _ in sorted(candidates, key=cmp_to_key(_cmp))
        ]

    def best_branch(
        self,
        *,
        child_value_candidate_getter: ChildValueCandidateGetter,
        semantic_compare: SemanticCompare,
    ) -> BranchKey | None:
        """Return the best child branch under current minimax decision semantics."""
        ordered = self.decision_ordered_branches(
            child_value_candidate_getter=child_value_candidate_getter,
            semantic_compare=semantic_compare,
        )
        if not ordered:
            return None
        return ordered[0]

    def second_best_branch(
        self,
        *,
        child_value_candidate_getter: ChildValueCandidateGetter,
        semantic_compare: SemanticCompare,
    ) -> BranchKey:
        """Return the current second-best child branch."""
        ordered = self.decision_ordered_branches(
            child_value_candidate_getter=child_value_candidate_getter,
            semantic_compare=semantic_compare,
        )
        assert len(ordered) >= 2
        return ordered[1]
