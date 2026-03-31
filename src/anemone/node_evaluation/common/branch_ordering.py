"""Provide branch-level decision-ordering helpers and protocols."""

from collections.abc import Callable, Iterable
from functools import cmp_to_key
from typing import Any, Protocol, runtime_checkable

from valanga import BranchKey
from valanga.evaluations import Value


class _Orderable(Protocol):
    def __lt__(self, other: Any, /) -> bool:
        """Return whether this tie-break value is less than another one."""
        ...

    def __gt__(self, other: Any, /) -> bool:
        """Return whether this tie-break value is greater than another one."""
        ...


def ordered_branches_from_candidates[T: _Orderable](
    candidates: Iterable[tuple[BranchKey, Value, T]],
    *,
    semantic_compare: Callable[[Value, Value], int],
) -> list[BranchKey]:
    """Derive branch order from semantic ``Value`` comparison plus tie-break data."""

    def _cmp(
        left: tuple[BranchKey, Value, T],
        right: tuple[BranchKey, Value, T],
    ) -> int:
        return -compare_branch_candidates(
            left_branch=left[0],
            left_value=left[1],
            left_tiebreak=left[2],
            right_branch=right[0],
            right_value=right[1],
            right_tiebreak=right[2],
            semantic_compare=semantic_compare,
        )

    return [branch_key for branch_key, _, _ in sorted(candidates, key=cmp_to_key(_cmp))]


def compare_branch_candidates[T: _Orderable](
    *,
    left_branch: BranchKey,
    left_value: Value,
    left_tiebreak: T,
    right_branch: BranchKey,
    right_value: Value,
    right_tiebreak: T,
    semantic_compare: Callable[[Value, Value], int],
) -> int:
    """Compare two branch candidates using the shared full-order semantics.

    Positive means the left branch is preferred, negative means the right branch
    is preferred, and zero means a full tie.
    """
    semantic = semantic_compare(left_value, right_value)
    if semantic != 0:
        return semantic
    if left_tiebreak < right_tiebreak:
        return 1
    if left_tiebreak > right_tiebreak:
        return -1
    if str(left_branch) < str(right_branch):
        return 1
    if str(left_branch) > str(right_branch):
        return -1
    return 0


@runtime_checkable
class DecisionOrderedEvaluation(Protocol):
    """Node-evaluation capability exposing derived branch decision ordering.

    This protocol is about branch order at one node. It does not describe
    ``Value``-level comparison policy or cached branch-ordering-key storage.
    """

    def decision_ordered_branches(self) -> list[BranchKey]:
        """Return child branches ordered by current decision semantics."""
        ...

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch key."""
        ...


@runtime_checkable
class SecondBestBranchAware(DecisionOrderedEvaluation, Protocol):
    """Node-evaluation capability exposing the current top-two branch order."""

    def second_best_branch(self) -> BranchKey:
        """Return the current second-best branch key."""
        ...


class SecondBestBranchCapabilityError(TypeError):
    """Raised when code expects top-two branch ordering from an evaluation."""

    def __init__(self, node_evaluation: object) -> None:
        """Initialize the error with the unsupported evaluation type."""
        super().__init__(
            f"Expected a SecondBestBranchAware evaluation, got {type(node_evaluation)!r}."
        )


def require_second_best_branch_aware(
    node_evaluation: object,
) -> SecondBestBranchAware:
    """Return a top-two-branch-aware evaluation or raise a clear type error."""
    if not isinstance(node_evaluation, SecondBestBranchAware):
        raise SecondBestBranchCapabilityError(node_evaluation)
    return node_evaluation
