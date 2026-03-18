"""Provide branch decision-ordering capabilities for tree-search evaluations."""

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
    """Return branches ordered by semantic value, tie-break data, then branch key."""

    def _cmp(
        left: tuple[BranchKey, Value, T],
        right: tuple[BranchKey, Value, T],
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

    return [branch_key for branch_key, _, _ in sorted(candidates, key=cmp_to_key(_cmp))]


@runtime_checkable
class DecisionOrderedEvaluation(Protocol):
    """Evaluation capability exposing family-specific child decision ordering."""

    def decision_ordered_branches(self) -> list[BranchKey]:
        """Return child branches ordered by current decision semantics."""
        ...

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch key."""
        ...


@runtime_checkable
class SecondBestBranchAware(DecisionOrderedEvaluation, Protocol):
    """Evaluation capability exposing the current top-two branch ordering."""

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
