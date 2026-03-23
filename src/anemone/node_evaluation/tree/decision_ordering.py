"""Provide cached branch-ordering keys plus branch-order derivation helpers."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from valanga import BranchKey
from valanga.evaluations import Value

from anemone.node_evaluation.common.branch_ordering import (
    ordered_branches_from_candidates,
)
from anemone.utils.my_value_sorted_dict import sort_dic


@dataclass(frozen=True, order=True, slots=True)
class BranchOrderingKey:
    """Explicit cached ordering key for one child branch at one node.

    Field order is intentionally the legacy tuple order so sorting and tie-break
    behavior remain unchanged:

    - ``primary_score``: objective-facing scalar used as the first ordering signal
    - ``tactical_tiebreak``: exact-line / tactical tie-break for equal scores
    - ``stable_tiebreak_id``: deterministic final tie-break, usually the child id
    """

    primary_score: float
    tactical_tiebreak: int
    stable_tiebreak_id: int


type BranchOrderingKeyGetter = Callable[[BranchKey], BranchOrderingKey]
type ChildValueCandidateGetter = Callable[[BranchKey], Value | None]
type SemanticCompare = Callable[[Value, Value], int]


def make_branch_ordering_keys_factory() -> dict[BranchKey, BranchOrderingKey]:
    """Create the cached branch-ordering-key storage."""
    return {}


@dataclass(slots=True)
class DecisionOrderingState:
    """Cache per-branch ordering keys and derive branch order from them.

    This class intentionally keeps two closely-related operational jobs together:
    it stores cached branch-ordering keys for one node, then derives branch order
    from current child ``Value`` candidates plus caller-supplied semantic
    comparison. It does not define ``Value`` semantics itself.
    """

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
        """Return the lowest cached branch-ordering key, if any."""
        if not self.branch_ordering_keys:
            return None
        return next(iter(self.branch_ordering_keys.values()))

    def decision_ordered_branches(
        self,
        *,
        child_value_candidate_getter: ChildValueCandidateGetter,
        semantic_compare: SemanticCompare,
    ) -> list[BranchKey]:
        """Derive child-branch order from semantic comparison plus cached keys."""
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
        """Return the best child branch under current semantic comparison."""
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
        """Return the current second-best child branch under decision ordering."""
        ordered = self.decision_ordered_branches(
            child_value_candidate_getter=child_value_candidate_getter,
            semantic_compare=semantic_compare,
        )
        assert len(ordered) >= 2
        return ordered[1]
