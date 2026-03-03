"""Explicit minimax backup policy with separated value/PV responsibilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from valanga import Color

from anemone.backup_policies.types import BackupResult

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
        NodeMinmaxEvaluation,
    )


class ExplicitMinimaxBackupPolicy:
    """Backup policy that computes float minimax value and PV with explicit invariants."""

    def backup_from_children(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Recompute value/PV from updated children and return change flags."""
        value_before_update = node_eval.get_value_white()
        best_branch_before_update = node_eval.best_branch()
        best_branch_sequence_before_update = node_eval.best_branch_sequence.copy()

        if branches_with_updated_value:
            node_eval.update_branches_values(
                branches_to_consider=branches_with_updated_value,
            )

        self._update_value_float_only(node_eval=node_eval)
        best_branch_after_update = node_eval.best_branch()

        if best_branch_after_update is not None:
            if node_eval.tree_node.all_branches_generated:
                pv_head_mismatch = (
                    not node_eval.best_branch_sequence
                    or node_eval.best_branch_sequence[0] != best_branch_after_update
                )
                best_child_pv_changed = (
                    best_branch_after_update in branches_with_updated_best_branch_seq
                )
                if (
                    best_branch_before_update != best_branch_after_update
                    or best_child_pv_changed
                    or pv_head_mismatch
                ):
                    node_eval.one_of_best_children_becomes_best_next_node()
            else:
                best_child = node_eval.tree_node.branches_children[
                    best_branch_after_update
                ]
                assert best_child is not None
                if node_eval.is_value_subjectively_better_than_evaluation(
                    best_child.tree_evaluation.get_value_white()
                ):
                    node_eval.one_of_best_children_becomes_best_next_node()
                else:
                    node_eval.best_branch_sequence = []
        elif node_eval.best_branch_sequence:
            node_eval.best_branch_sequence = []

        if (
            branches_with_updated_best_branch_seq
            and best_branch_after_update is not None
            and best_branch_after_update in branches_with_updated_best_branch_seq
            and node_eval.best_branch_sequence
        ):
            node_eval.update_best_branch_sequence(
                branches_with_updated_best_branch_seq
            )

        value_changed = value_before_update != node_eval.get_value_white()
        pv_changed = node_eval.best_branch_sequence != best_branch_sequence_before_update

        return BackupResult(
            value_changed=value_changed,
            pv_changed=pv_changed,
            over_changed=False,
        )

    def _update_value_float_only(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
    ) -> None:
        """Update ``value_white_minmax`` using only float bridge values."""
        best_branch_key = node_eval.best_branch()
        assert best_branch_key is not None

        best_child = node_eval.tree_node.branches_children[best_branch_key]
        assert best_child is not None
        best_child_value_white = best_child.tree_evaluation.get_value_white()

        if node_eval.tree_node.all_branches_generated:
            value_after_update = best_child_value_white
        else:
            direct_evaluation = node_eval.value_white_direct_evaluation
            assert direct_evaluation is not None
            if node_eval.tree_node.state.turn is Color.WHITE:
                value_after_update = max(best_child_value_white, direct_evaluation)
            else:
                value_after_update = min(best_child_value_white, direct_evaluation)

        node_eval.value_white_minmax = value_after_update
