"""Explicit max backup policy for single-agent node evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.backup_policies.common import SelectedValue, all_child_values_exact
from anemone.backup_policies.explicit_minimax import has_value_changed
from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation import canonical_value

if TYPE_CHECKING:
    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone.node_evaluation.node_max_evaluation import NodeMaxEvaluation


class ExplicitMaxBackupPolicy:
    """Backup policy that computes single-agent max value and PV from Value."""

    def backup_from_children(
        self,
        node_eval: NodeMaxEvaluation[Any],
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Recompute backed-up value and PV from child values."""
        del branches_with_updated_value

        value_before = node_eval.backed_up_value
        pv_before = node_eval.best_branch_sequence.copy()
        over_before = node_eval.over_event

        best_branch_key = node_eval.best_branch()
        best_child_value = (
            node_eval.child_value_candidate(best_branch_key)
            if best_branch_key is not None
            else None
        )
        direct_value = node_eval.direct_value

        selection = self._select_value(
            node_eval=node_eval,
            best_child_value=best_child_value,
            direct_value=direct_value,
        )
        value_after = self._build_backed_up_value(
            node_eval=node_eval,
            selection=selection,
        )
        node_eval.backed_up_value = value_after

        pv_changed = self._update_pv(
            node_eval=node_eval,
            best_branch_key=best_branch_key,
            best_child_value=best_child_value,
            selection=selection,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )

        return BackupResult(
            value_changed=has_value_changed(
                value_before=value_before,
                value_after=node_eval.backed_up_value,
            ),
            pv_changed=pv_changed or pv_before != node_eval.best_branch_sequence,
            over_changed=over_before != node_eval.over_event,
        )

    def _select_value(
        self,
        *,
        node_eval: NodeMaxEvaluation[Any],
        best_child_value: Value | None,
        direct_value: Value | None,
    ) -> SelectedValue:
        if best_child_value is None:
            return SelectedValue(value=direct_value, from_child=False)
        if direct_value is None:
            return SelectedValue(value=best_child_value, from_child=True)
        if node_eval.tree_node.all_branches_generated:
            return SelectedValue(value=best_child_value, from_child=True)
        if (
            node_eval.objective.semantic_compare(
                best_child_value,
                direct_value,
                node_eval.tree_node.state,
            )
            >= 0
        ):
            return SelectedValue(value=best_child_value, from_child=True)
        return SelectedValue(value=direct_value, from_child=False)

    def _build_backed_up_value(
        self,
        *,
        node_eval: NodeMaxEvaluation[Any],
        selection: SelectedValue,
    ) -> Value | None:  # pylint: disable=duplicate-code
        chosen_value = selection.value  # pylint: disable=duplicate-code
        if chosen_value is None:
            return None

        if not selection.from_child:
            return chosen_value

        exact_from_children = (
            node_eval.tree_node.all_branches_generated
            and all_child_values_exact(node_eval)
        )
        return canonical_value.make_backed_up_value(
            score=chosen_value.score,
            exact=exact_from_children,
            node_is_terminal=False,
            over_event=chosen_value.over_event if exact_from_children else None,
        )

    def _update_pv(
        self,
        *,
        node_eval: NodeMaxEvaluation[Any],
        best_branch_key: BranchKey | None,
        best_child_value: Value | None,
        selection: SelectedValue,
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> bool:
        if (
            best_branch_key is None
            or best_child_value is None
            or not selection.from_child
            or selection.value != best_child_value
        ):
            return node_eval.set_best_branch_sequence([])

        best_child = node_eval.tree_node.branches_children[best_branch_key]
        assert best_child is not None
        new_sequence = [
            best_branch_key,
            *best_child.tree_evaluation.best_branch_sequence,
        ]
        if (
            not node_eval.best_branch_sequence
            or node_eval.best_branch_sequence[0] != best_branch_key
            or best_branch_key in branches_with_updated_best_branch_seq
        ):
            return node_eval.set_best_branch_sequence(new_sequence)
        return False
