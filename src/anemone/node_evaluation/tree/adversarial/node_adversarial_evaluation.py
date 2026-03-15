"""Provide the adversarial node-evaluation protocol."""

from typing import Protocol

from valanga import (
    BranchKey,
    State,
)
from valanga.evaluations import Value

from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.common.node_value_evaluation import NodeValueEvaluation


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

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch key."""
        ...

    def second_best_branch(self) -> BranchKey:
        """Return the second-best branch key."""
        ...
