"""Explicit minimax backup policy with separated value/PV responsibilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from valanga.evaluations import Certainty

from anemone.backup_policies.aggregation import (
    AggregationPolicy,
    MinimaxAggregationPolicy,
)
from anemone.backup_policies.common import (
    ProofClassification,
    SelectedValue,
    all_child_values_exact,
    finalize_selection_with_proof,
)
from anemone.node_evaluation.common import canonical_value

if TYPE_CHECKING:
    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone.backup_policies.types import BackupResult
    from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
        NodeMinmaxEvaluation,
    )


class ExplicitMinimaxBackupPolicy:
    """Backup policy that computes adversarial minimax value/PV/over from Value."""

    aggregation_policy: AggregationPolicy[NodeMinmaxEvaluation[Any, Any]]

    def __init__(
        self,
        aggregation_policy: AggregationPolicy[NodeMinmaxEvaluation[Any, Any]]
        | None = None,
    ) -> None:
        """Initialize the backup policy with one injectable aggregation strategy."""
        if aggregation_policy is None:
            aggregation_policy = MinimaxAggregationPolicy()
        self.aggregation_policy = aggregation_policy

    def backup_from_children(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Recompute value/PV/over from updated children and return change flags."""
        assert (
            not node_eval.tree_node.all_branches_generated
            or node_eval.tree_node.branches_children
        ), "Cannot compute minimax value: no children."

        best_branch_before_update = node_eval.best_branch()
        selection = self.aggregation_policy.select_value(
            node_eval=node_eval,
            branches_with_updated_value=branches_with_updated_value,
        )
        proof = self._classify_selected_value(
            node_eval=node_eval,
            selection=selection,
        )
        return finalize_selection_with_proof(
            node_eval=node_eval,
            selection=selection,
            proof=proof,
            branches_with_updated_value=branches_with_updated_value,
            update_pv=lambda: self._refresh_pv(
                node_eval=node_eval,
                best_branch_before_update=best_branch_before_update,
                selection=selection,
                branches_with_updated_best_branch_seq=(
                    branches_with_updated_best_branch_seq
                ),
            ),
        )

    def _classify_selected_value(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        selection: SelectedValue,
    ) -> ProofClassification | None:
        """Classify certainty/outcome for the selected parent value."""
        chosen_value = selection.value
        if chosen_value is None:
            return None

        if not selection.from_child:
            return ProofClassification.from_value(chosen_value)

        exact = self._selected_child_proves_exact_value(
            node_eval=node_eval,
            selected_child_value=chosen_value,
        )
        return ProofClassification(
            certainty=Certainty.FORCED if exact else Certainty.ESTIMATE,
            over_event=chosen_value.over_event if exact else None,
        )

    def _selected_child_proves_exact_value(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        selected_child_value: Value,
    ) -> bool:
        if not canonical_value.is_exact_value(selected_child_value):
            return False

        # Important: child certainty must never be copied directly to the parent.
        # A non-terminal parent becomes FORCED only when the parent itself is now exact.
        if self._selected_child_proves_parent_exact_value_by_choice(
            node_eval=node_eval,
            selected_child_value=selected_child_value,
        ):
            return True

        if not node_eval.tree_node.all_branches_generated:
            return False

        return all_child_values_exact(node_eval)

    def _selected_child_proves_parent_exact_value_by_choice(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        selected_child_value: Value,
    ) -> bool:
        """Return True when the chosen child alone proves the parent's exact value.

        A current-player exact win is enough to solve the parent even before all
        siblings are generated, because the player to move can choose that line.
        """
        over_event = selected_child_value.over_event
        return over_event is not None and over_event.is_winner(
            node_eval.tree_node.state.turn
        )

    def _refresh_pv(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        best_branch_before_update: BranchKey | None,
        selection: SelectedValue,
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> bool:
        """Refresh minimax PV after the shared backed-up-value update is applied."""
        best_branch_after_update = node_eval.best_branch()
        if best_branch_after_update is None or not selection.from_child:
            return node_eval.clear_best_branch_sequence()

        pv_changed = False
        if self._should_rebuild_pv_from_best_branch(
            node_eval=node_eval,
            best_branch_before_update=best_branch_before_update,
            best_branch_after_update=best_branch_after_update,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        ):
            pv_changed = node_eval.one_of_best_children_becomes_best_next_node()

        if (
            node_eval.best_branch_sequence
            and node_eval.best_branch_sequence[0] == best_branch_after_update
            and branches_with_updated_best_branch_seq
        ):
            pv_changed = (
                node_eval.update_best_branch_sequence(
                    branches_with_updated_best_branch_seq
                )
                or pv_changed
            )
        return pv_changed

    def _should_rebuild_pv_from_best_branch(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        best_branch_before_update: BranchKey | None,
        best_branch_after_update: BranchKey,
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> bool:
        """Return whether minimax should rebuild the PV head from the best child."""
        return (
            best_branch_before_update != best_branch_after_update
            or not node_eval.best_branch_sequence
            or node_eval.best_branch_sequence[0] != best_branch_after_update
            or best_branch_after_update in branches_with_updated_best_branch_seq
        )
