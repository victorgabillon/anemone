"""Provide branch decision-ordering capabilities for tree-search evaluations."""

from typing import Protocol, runtime_checkable

from valanga import BranchKey


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
