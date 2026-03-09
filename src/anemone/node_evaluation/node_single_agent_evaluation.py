"""Provide the single-agent node-evaluation protocol."""

from typing import Protocol

from valanga import BranchKey, State

from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.node_value_evaluation import NodeValueEvaluation


class NodeSingleAgentEvaluation[StateT: State = State](NodeValueEvaluation, Protocol):
    """Single-agent decision and backup semantics layered on canonical values.

    The canonical Value access surface is inherited from ``NodeValueEvaluation``.
    This protocol adds the family-specific decision and backup operations.
    """

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
        """Run single-agent backup after child updates."""
        ...

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch key."""
        ...
