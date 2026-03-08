"""Provide the NodeTreeEvaluation interface."""

from typing import Protocol

from valanga import (
    BranchKey,
    State,
    StateEvaluation,
)

from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.node_value_evaluation import NodeValueEvaluation
from anemone.values import Value


class NodeTreeEvaluation[StateT: State = State](NodeValueEvaluation, Protocol):
    """Interface for node tree evaluation.

    This is the evaluation of a node that is based both on a direct evaluation of the state within
    the NodeTreeEvaluation and its children.
    The direct evaluation is used to evaluate leaf nodes, while the children evaluations are used to propagate values up the tree.

    """

    @property
    def direct_value(self) -> Value | None:
        """Return the canonical direct evaluation value for this node."""
        ...

    @direct_value.setter
    def direct_value(self, value: Value | None) -> None:
        """Set the canonical direct evaluation value for this node."""
        ...

    @property
    def backed_up_value(self) -> Value | None:
        """Return the canonical backed-up value for this node."""
        ...

    @backed_up_value.setter
    def backed_up_value(self, value: Value | None) -> None:
        """Set the canonical backed-up value for this node."""
        ...

    @property
    def minmax_value(self) -> Value | None:
        """Return the minimax-specific backed-up value for this node."""
        ...

    @minmax_value.setter
    def minmax_value(self, value: Value | None) -> None:
        """Set the minimax-specific backed-up value for this node."""
        ...

    best_branch_sequence: list[BranchKey]

    def update_best_branch_sequence(
        self, branches_with_updated_best_branch_seq: set[BranchKey]
    ) -> bool:
        """Update the best branch sequence from updated branches."""
        ...

    def backup_from_children(
        self,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Run backup policy from updated children and return changed-state flags."""
        ...

    def update_over(self, branches_with_updated_over: set[BranchKey]) -> bool:
        """Update terminal state based on updated branches."""
        ...

    def evaluate(self) -> StateEvaluation:
        """Return a state evaluation for this node."""
        ...

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch key."""
        ...

    def second_best_branch(self) -> BranchKey:
        """Return the second-best branch key."""
        ...
