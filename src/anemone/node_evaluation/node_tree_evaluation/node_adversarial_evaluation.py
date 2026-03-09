"""Provide the adversarial node-evaluation protocol."""

from typing import Protocol

from valanga import (
    BranchKey,
    State,
    StateEvaluation,
)

from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.node_value_evaluation import NodeValueEvaluation
from anemone.values import Value


class NodeAdversarialEvaluation[StateT: State = State](NodeValueEvaluation, Protocol):
    """Adversarial decision and backup semantics layered on canonical values."""

    @property
    def minmax_value(self) -> Value | None:
        """Return the adversarial backed-up minimax value for this node."""
        ...

    @minmax_value.setter
    def minmax_value(self, value: Value | None) -> None:
        """Set the adversarial backed-up minimax value for this node."""
        ...

    best_branch_sequence: list[BranchKey]

    def update_best_branch_sequence(
        self, branches_with_updated_best_branch_seq: set[BranchKey]
    ) -> bool:
        """Update the principal variation from changed child branches."""
        ...

    def backup_from_children(
        self,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Run adversarial backup after child updates."""
        ...

    def update_over(self, branches_with_updated_over: set[BranchKey]) -> bool:
        """Update terminal-state information from changed child branches."""
        ...

    def evaluate(self) -> StateEvaluation:
        """Return a state evaluation derived from adversarial node semantics."""
        ...

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch key."""
        ...

    def second_best_branch(self) -> BranchKey:
        """Return the second-best branch key."""
        ...
