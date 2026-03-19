"""Explicit max backup policy for single-agent node evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.backup_policies.aggregation import (
    AggregationPolicy,
    MaxAggregationPolicy,
)
from anemone.backup_policies.common import (
    SelectedValue,
    finalize_selection_with_proof,
)
from anemone.backup_policies.proof import MaxProofPolicy, ProofPolicy

if TYPE_CHECKING:
    from valanga import BranchKey
    from valanga.evaluations import Value

    from anemone.backup_policies.types import BackupResult
    from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
        NodeMaxEvaluation,
    )


class ExplicitMaxBackupPolicy:
    """Backup policy that computes single-agent max value and PV from Value."""

    aggregation_policy: AggregationPolicy[NodeMaxEvaluation[Any]]
    proof_policy: ProofPolicy[NodeMaxEvaluation[Any]]

    def __init__(
        self,
        aggregation_policy: AggregationPolicy[NodeMaxEvaluation[Any]] | None = None,
        proof_policy: ProofPolicy[NodeMaxEvaluation[Any]] | None = None,
    ) -> None:
        """Initialize the backup policy with injectable aggregation/proof strategies."""
        if aggregation_policy is None:
            aggregation_policy = MaxAggregationPolicy()
        if proof_policy is None:
            proof_policy = MaxProofPolicy()
        self.aggregation_policy = aggregation_policy
        self.proof_policy = proof_policy

    def backup_from_children(
        self,
        node_eval: NodeMaxEvaluation[Any],
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Recompute backed-up value and PV from child values."""
        # pylint: disable=duplicate-code
        best_branch_key = node_eval.best_branch()
        best_child_value = (
            node_eval.child_value_candidate(best_branch_key)
            if best_branch_key is not None
            else None
        )

        selection = self.aggregation_policy.select_value(
            node_eval=node_eval,
            branches_with_updated_value=branches_with_updated_value,
        )
        proof = self.proof_policy.classify_selected_value(
            node_eval=node_eval,
            selection=selection,
        )
        return finalize_selection_with_proof(
            node_eval=node_eval,
            selection=selection,
            proof=proof,
            branches_with_updated_value=branches_with_updated_value,
            update_pv=lambda: self._update_pv(
                node_eval=node_eval,
                best_branch_key=best_branch_key,
                best_child_value=best_child_value,
                selection=selection,
                branches_with_updated_best_branch_seq=(
                    branches_with_updated_best_branch_seq
                ),
            ),
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
            return node_eval.clear_best_branch_sequence()

        new_sequence = node_eval.best_branch_line_from_child(best_branch_key)
        if (
            not node_eval.best_branch_sequence
            or node_eval.best_branch_sequence[0] != best_branch_key
            or best_branch_key in branches_with_updated_best_branch_seq
        ):
            return node_eval.set_best_branch_sequence(new_sequence)
        return False
