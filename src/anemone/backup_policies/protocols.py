"""Protocols for backup-policy implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.backup_policies.types import BackupResult


class BackupPolicy[NodeEvalT](Protocol):
    """Protocol for orchestrating family-specific node backups from updated children."""

    def backup_from_children(
        self,
        node_eval: NodeEvalT,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Perform subtree backup operations and return family-neutral change flags."""
        ...
