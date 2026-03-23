"""Protocols for backup-policy implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from anemone.backup_policies.common import BackupPipelineNode

if TYPE_CHECKING:
    from valanga import BranchKey
    from valanga.evaluations import Value

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


class ExplicitTreeBackupNodeEval(BackupPipelineNode, Protocol):
    """Minimal node-evaluation surface needed by the shared explicit backup engine."""

    def best_branch(self) -> BranchKey | None:
        """Return the current best child branch."""
        ...

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the current candidate value for one child branch."""
        ...

    def update_branches_values(self, branches_to_consider: set[BranchKey]) -> None:
        """Refresh branch-ordering keys for the provided child branches."""
        ...

    def clear_best_branch_sequence(self) -> bool:
        """Clear the stored principal variation."""
        ...

    def update_best_branch_sequence(
        self,
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> bool:
        """Refresh the stored principal variation tail from updated children."""
        ...

    def one_of_best_children_becomes_best_next_node(self) -> bool:
        """Rebuild the principal variation from the current best child."""
        ...
