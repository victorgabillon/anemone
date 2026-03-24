"""Branch-ordering helpers extracted from the generic tree-evaluation engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Protocol, cast

from valanga import BranchKey

from anemone.node_evaluation.tree.decision_ordering import (
    BranchOrderingKey,
    DecisionOrderingState,
)

if TYPE_CHECKING:
    from valanga.evaluations import Value


type ChildValueCandidateGetter = Callable[[BranchKey], Value | None]
type TacticalQualityGetter = Callable[[BranchKey], int]
type PrimaryScoreGetter = Callable[[BranchKey], float]
type BranchOrderingKeyGetter = Callable[[BranchKey], BranchOrderingKey]
type ChildNodeGetter = Callable[[BranchKey], object | None]


class StableIdChild(Protocol):
    """Minimal child-node surface needed for stable branch ordering."""

    id: int


def _missing_branch_child_error(branch_key: BranchKey) -> RuntimeError:
    return RuntimeError(
        "Cannot compute branch-ordering key for branch "
        f"{branch_key!r}: no child node is linked to it."
    )


def _missing_branch_value_error(branch_key: BranchKey) -> RuntimeError:
    return RuntimeError(
        "Cannot compute branch-ordering key for branch "
        f"{branch_key!r}: no Value is available yet. "
        "Evaluate or back up the child before ordering branches."
    )


def ordered_candidate_branches_with_child_fallback(
    *,
    preferred_ordered_branches: Iterable[BranchKey],
    available_child_branches: Mapping[BranchKey, object],
) -> tuple[BranchKey, ...]:
    """Return preferred branches first, then remaining child branches."""
    ordered_branches: list[BranchKey] = []
    seen_branches: set[BranchKey] = set()

    for branch in preferred_ordered_branches:
        if branch in seen_branches:
            continue
        if branch not in available_child_branches:
            continue
        ordered_branches.append(branch)
        seen_branches.add(branch)

    for branch in available_child_branches:
        if branch in seen_branches:
            continue
        ordered_branches.append(branch)
        seen_branches.add(branch)

    return tuple(ordered_branches)


def branches_with_ordering_key_available(
    *,
    branches_to_consider: set[BranchKey],
    child_value_candidate_getter: ChildValueCandidateGetter,
) -> set[BranchKey]:
    """Return branches that currently have enough data to be ordered."""
    return {
        branch_key
        for branch_key in branches_to_consider
        if child_value_candidate_getter(branch_key) is not None
    }


def compute_branch_ordering_key(
    *,
    branch_key: BranchKey,
    child_node_getter: ChildNodeGetter,
    child_value_candidate_getter: ChildValueCandidateGetter,
    primary_score_getter: PrimaryScoreGetter,
    tactical_quality_getter: TacticalQualityGetter,
) -> BranchOrderingKey:
    """Build the cached branch-ordering key for one child branch."""
    child = cast("StableIdChild | None", child_node_getter(branch_key))
    if child is None:
        raise _missing_branch_child_error(branch_key)

    child_value = child_value_candidate_getter(branch_key)
    if child_value is None:
        raise _missing_branch_value_error(branch_key)
    del child_value

    return BranchOrderingKey(
        primary_score=primary_score_getter(branch_key),
        tactical_tiebreak=tactical_quality_getter(branch_key),
        stable_tiebreak_id=child.id,
    )


def update_branches_values(
    *,
    decision_ordering: DecisionOrderingState,
    branches_to_consider: set[BranchKey],
    child_value_candidate_getter: ChildValueCandidateGetter,
    branch_ordering_key_getter: BranchOrderingKeyGetter,
) -> None:
    """Refresh cached ordering keys for branches that currently have a value."""
    decision_ordering.update_branch_ordering_keys(
        branches_with_ordering_key_available(
            branches_to_consider=branches_to_consider,
            child_value_candidate_getter=child_value_candidate_getter,
        ),
        branch_ordering_key_getter=branch_ordering_key_getter,
    )
