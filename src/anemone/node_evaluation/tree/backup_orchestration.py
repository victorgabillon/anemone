"""Thin backup-policy adapter for generic tree-evaluation families."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.backup_policies.protocols import BackupPolicy
    from anemone.backup_policies.types import BackupResult


def backup_from_children(
    *,
    backup_policy: BackupPolicy[Any] | None,
    node_eval: object,
    branches_with_updated_value: set[BranchKey],
    branches_with_updated_best_branch_seq: set[BranchKey],
) -> BackupResult:
    """Delegate backup work to the configured tree-evaluation backup policy."""
    assert backup_policy is not None
    return backup_policy.backup_from_children(
        node_eval=node_eval,
        branches_with_updated_value=branches_with_updated_value,
        branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
    )
