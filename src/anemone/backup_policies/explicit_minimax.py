"""Explicit minimax backup policy with separated value/PV responsibilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from valanga.evaluations import Certainty

from anemone.backup_policies.common import (
    ProofClassification,
    SelectedValue,
    all_child_values_exact,
    finalize_selection_with_proof,
    select_value_from_best_child_and_direct,
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

        if branches_with_updated_value:
            self._update_branches_values_value_only(
                node_eval=node_eval,
                branches_to_consider=branches_with_updated_value,
            )

        selection = self._select_value(node_eval=node_eval)
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

    def _select_value(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
    ) -> SelectedValue:
        """Choose the generic backed-up-value candidate for this minimax node."""
        best_child_value = self._best_child_value_for_selection(node_eval=node_eval)
        if best_child_value is None:
            assert node_eval.tree_node.branches_children, (
                "Cannot compute minimax value: no children."
            )
            direct_value = self._direct_value_candidate(node_eval)
            assert direct_value is not None, (
                "Explicit minimax requires direct_value for partially expanded nodes."
            )
            return SelectedValue(value=direct_value, from_child=False)

        direct_value = self._direct_value_candidate(node_eval)
        if not node_eval.tree_node.all_branches_generated:
            assert direct_value is not None, (
                "Explicit minimax requires direct_value for partially expanded nodes."
            )

        return select_value_from_best_child_and_direct(
            best_child_value=best_child_value,
            direct_value=direct_value,
            all_branches_generated=node_eval.tree_node.all_branches_generated,
            child_beats_direct=lambda child, direct: (
                node_eval.child_is_better_than_direct(
                    child,
                    direct,
                    side_to_move=node_eval.tree_node.state.turn,
                )
            ),
        )

    def _best_child_value_for_selection(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
    ) -> Value | None:
        """Return the current best child candidate, refreshing ordering if needed."""
        best_branch_key = node_eval.best_branch()
        if best_branch_key is None and node_eval.tree_node.branches_children:
            self._update_branches_values_value_only(
                node_eval=node_eval,
                branches_to_consider=set(node_eval.tree_node.branches_children),
            )
            best_branch_key = node_eval.best_branch()

        if best_branch_key is None:
            return None

        best_child_value = node_eval.child_value_candidate(best_branch_key)
        assert best_child_value is not None
        return best_child_value

    def _update_branches_values_value_only(
        self,
        *,
        node_eval: NodeMinmaxEvaluation[Any, Any],
        branches_to_consider: set[BranchKey],
    ) -> None:
        """Update branch ordering keys using Value semantics instead of float-only keys."""
        node_eval.update_branches_values(
            {
                branch_key
                for branch_key in branches_to_consider
                if node_eval.child_value_candidate(branch_key) is not None
            }
        )

    def _direct_value_candidate(
        self,
        node_eval: NodeMinmaxEvaluation[Any, Any],
    ) -> Value | None:
        """Return the local direct-evaluator candidate used in partial expansion."""
        return node_eval.direct_value

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
