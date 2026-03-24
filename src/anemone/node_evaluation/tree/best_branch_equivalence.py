"""Pure best-branch equivalence helpers for the generic tree-evaluation engine."""

from __future__ import annotations

from collections.abc import Callable
from math import log
from typing import TYPE_CHECKING

from valanga import BranchKey

if TYPE_CHECKING:
    from valanga.evaluations import Value


type ChildValueCandidateGetter = Callable[[BranchKey], Value | None]
type SemanticCompare = Callable[[Value, Value], int]
type TacticalQualityKeyGetter = Callable[[BranchKey], object]


def branch_values_are_equal(
    *,
    branch: BranchKey,
    best_branch: BranchKey,
    child_value_candidate_getter: ChildValueCandidateGetter,
    tactical_quality_key_getter: TacticalQualityKeyGetter,
) -> bool:
    """Return whether two branches share value and tactical quality."""
    branch_value = child_value_candidate_getter(branch)
    best_value = child_value_candidate_getter(best_branch)
    assert branch_value is not None
    assert best_value is not None
    return branch_value == best_value and tactical_quality_key_getter(
        branch
    ) == tactical_quality_key_getter(best_branch)


def branch_values_are_considered_equal(
    *,
    branch: BranchKey,
    best_branch: BranchKey,
    child_value_candidate_getter: ChildValueCandidateGetter,
    semantic_compare: SemanticCompare,
) -> bool:
    """Return whether two branches tie under the node's semantic comparison."""
    branch_value = child_value_candidate_getter(branch)
    best_value = child_value_candidate_getter(best_branch)
    assert branch_value is not None
    assert best_value is not None
    return semantic_compare(branch_value, best_value) == 0


def are_almost_equal_scores(left: float, right: float) -> bool:
    """Return whether two scalar branch-equivalence scores are close enough."""
    epsilon = 0.01
    return left > right - epsilon and right > left - epsilon


def logistic_equivalence_score(x: float) -> float:
    """Apply the shared logistic-style transform used for branch equivalence."""
    y = min(max(x * 0.5 + 0.5, 0.000000000000000000000001), 0.9999999999999999)
    return log(y / (1 - y)) * max(1, abs(x))
