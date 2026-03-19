"""Provide shared decision-ordering bookkeeping for tree-evaluation families."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from valanga import BranchKey
from valanga.evaluations import Value

from anemone.node_evaluation.common.branch_ordering import (
    ordered_branches_from_candidates,
)
from anemone.utils.my_value_sorted_dict import sort_dic

type BranchOrderingKey = tuple[float, int, int]
type BranchOrderingKeyGetter = Callable[[BranchKey], BranchOrderingKey]
type ChildValueCandidateGetter = Callable[[BranchKey], Value | None]
type SemanticCompare = Callable[[Value, Value], int]


def make_branch_ordering_keys_factory() -> dict[BranchKey, BranchOrderingKey]:
    """Create the cached branch-ordering-key storage."""
    return {}


@dataclass(slots=True)
class DecisionOrderingState:
    """Store cached branch-ordering keys and derive decision ordering from them."""

    branch_ordering_keys: dict[BranchKey, BranchOrderingKey] = field(
        default_factory=make_branch_ordering_keys_factory
    )

    def record_ordering_key(
        self,
        branch_key: BranchKey,
        *,
        branch_ordering_key_getter: BranchOrderingKeyGetter,
    ) -> None:
        """Refresh the cached ordering key for one child branch."""
        self.branch_ordering_keys[branch_key] = branch_ordering_key_getter(branch_key)

    def update_branch_ordering_keys(
        self,
        branches_to_consider: Iterable[BranchKey],
        *,
        branch_ordering_key_getter: BranchOrderingKeyGetter,
    ) -> None:
        """Refresh cached ordering keys for the provided branches."""
        for branch_key in branches_to_consider:
            self.record_ordering_key(
                branch_key,
                branch_ordering_key_getter=branch_ordering_key_getter,
            )
        self.branch_ordering_keys = sort_dic(self.branch_ordering_keys)

    def best_branch_value(self) -> BranchOrderingKey | None:
        """Return the cached ordering key of the current best search branch."""
        if not self.branch_ordering_keys:
            return None
        return next(iter(self.branch_ordering_keys.values()))

    def decision_ordered_branches(
        self,
        *,
        child_value_candidate_getter: ChildValueCandidateGetter,
        semantic_compare: SemanticCompare,
    ) -> list[BranchKey]:
        """Return child branches ordered by the caller's decision semantics."""
        candidates: list[tuple[BranchKey, Value, BranchOrderingKey]] = []
        for branch_key, ordering_key in self.branch_ordering_keys.items():
            child_value = child_value_candidate_getter(branch_key)
            if child_value is None:
                continue
            candidates.append((branch_key, child_value, ordering_key))
        return ordered_branches_from_candidates(
            candidates,
            semantic_compare=semantic_compare,
        )

    def best_branch(
        self,
        *,
        child_value_candidate_getter: ChildValueCandidateGetter,
        semantic_compare: SemanticCompare,
    ) -> BranchKey | None:
        """Return the best child branch under current decision semantics."""
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
