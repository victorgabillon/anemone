"""Provide the generic tree-search node-evaluation protocol."""

from typing import Protocol

from valanga import BranchKey, State

from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.common.node_value_evaluation import NodeValueEvaluation


class NodeTreeEvaluation[StateT: State = State](NodeValueEvaluation, Protocol):
    """Shared tree-search evaluation surface used by generic search orchestration."""

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
        """Run family-specific backup after child updates."""
        ...

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch key."""
        ...
