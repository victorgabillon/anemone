"""Legacy backup policy reproducing current minimax backup behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.backup_policies.types import BackupResult

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
        NodeMinmaxEvaluation,
    )


class LegacyMinimaxBackupPolicy:
    """Thin adapter policy that delegates to existing NodeMinmaxEvaluation methods."""

    def backup_from_children(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Run legacy backup sequence and return behavior flags."""
        over_before_update = node_eval.over_event
        value_changed, pv_changed_from_values = (
            node_eval.minmax_value_update_from_children_legacy(
                branches_with_updated_value=branches_with_updated_value
            )
        )
        node_eval.sync_over_from_values()

        pv_changed_from_sequence = False
        if branches_with_updated_best_branch_seq:
            pv_changed_from_sequence = node_eval.update_best_branch_sequence(
                branches_with_updated_best_branch_seq
            )

        return BackupResult(
            value_changed=value_changed,
            pv_changed=pv_changed_from_values or pv_changed_from_sequence,
            over_changed=over_before_update != node_eval.over_event,
        )
