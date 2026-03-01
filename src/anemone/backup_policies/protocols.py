"""Protocols for backup-policy implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.backup_policies.types import BackupResult
    from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
        NodeMinmaxEvaluation,
    )


class BackupPolicy(Protocol):
    """Protocol for orchestrating node backups from updated children."""

    def backup_from_children(
        self,
        node_eval: NodeMinmaxEvaluation,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Perform backup operations and return changed-state flags."""
        ...
