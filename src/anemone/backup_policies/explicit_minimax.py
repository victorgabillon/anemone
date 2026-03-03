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
        pv_before_update = node_eval.best_branch_sequence.copy()
        best_branch_before_update = node_eval.best_branch()

        if branches_with_updated_value:
            node_eval.update_branches_values(
                branches_to_consider=branches_with_updated_value,
            )

        self._update_value_float_only(node_eval=node_eval)
        best_branch_after_update = node_eval.best_branch()

        if best_branch_after_update is None:
            node_eval._set_best_branch_sequence([])
        else:
            pv_head_mismatch = (
                not node_eval.best_branch_sequence
                or node_eval.best_branch_sequence[0] != best_branch_after_update
            )
            best_branch_changed = best_branch_before_update != best_branch_after_update
            best_child_pv_updated = (
                best_branch_after_update in branches_with_updated_best_branch_seq
            )

            best_child = None
            best_child_version_changed = False
            if not pv_head_mismatch:
                best_child = node_eval.tree_node.branches_children[best_branch_after_update]
                assert best_child is not None
                best_child_version = int(
                    getattr(best_child.tree_evaluation, "pv_version", 0)
                )
                best_child_version_changed = (
                    node_eval._pv_cached_best_child_version != best_child_version
                )

            should_rebuild = (
                best_branch_changed
                or pv_head_mismatch
                or best_child_version_changed
                or best_child_pv_updated
            )

            did_rebuild = False
            if node_eval.tree_node.all_branches_generated:
                if should_rebuild:
                    did_rebuild = node_eval.one_of_best_children_becomes_best_next_node()
            else:
                direct_evaluation = node_eval.value_white_direct_evaluation
                if direct_evaluation is None:
                    node_eval._set_best_branch_sequence([])
                else:
                    if best_child is None:
                        best_child = node_eval.tree_node.branches_children[
                            best_branch_after_update
                        ]
                        assert best_child is not None
                    if self._is_child_better_than_direct_evaluation(
                        node_eval=node_eval,
                        child_value_white=best_child.tree_evaluation.get_value_white(),
                    ):
                        if should_rebuild:
                            did_rebuild = (
                                node_eval.one_of_best_children_becomes_best_next_node()
                            )
                    else:
                        node_eval._set_best_branch_sequence([])

            if (
                node_eval.best_branch_sequence
                and best_child_pv_updated
                and not did_rebuild
            ):
                node_eval.update_best_branch_sequence(
                    branches_with_updated_best_branch_seq
                )

        value_changed = value_before_update != node_eval.get_value_white()
        pv_changed = pv_before_update != node_eval.best_branch_sequence

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

    def _is_child_better_than_direct_evaluation(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        child_value_white: float,
    ) -> bool:
        """Return whether child value dominates direct eval for the side to move."""
        direct_evaluation = node_eval.value_white_direct_evaluation
        assert direct_evaluation is not None
        if node_eval.tree_node.state.turn is Color.WHITE:
            return child_value_white >= direct_evaluation
        return child_value_white <= direct_evaluation
