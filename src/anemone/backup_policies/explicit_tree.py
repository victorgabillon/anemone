"""Shared explicit tree-backup engine used by named family wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.backup_policies.aggregation import (
    AggregationPolicy,
    BestChildAggregationPolicy,
    prepare_best_child_aggregation,
)
from anemone.backup_policies.common import (
    SelectedValue,
    finalize_selection_with_proof,
)
from anemone.backup_policies.protocols import ExplicitTreeBackupNodeEval

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.backup_policies.proof import ProofPolicy
    from anemone.backup_policies.types import BackupResult


class ExplicitTreeBackupPolicy[NodeEvalT: ExplicitTreeBackupNodeEval]:
    """Run the shared explicit tree-backup flow for one tree-evaluation family.

    Aggregation computes a child/tree-derived candidate for ``backed_up_value``.
    Canonical-value helpers still decide that ``direct_value`` remains canonical
    whenever no backed-up candidate exists. Family-specific wrappers such as
    ``ExplicitMinimaxBackupPolicy`` and ``ExplicitMaxBackupPolicy`` exist to keep
    readable public names while only installing family defaults here.
    """

    aggregation_policy: AggregationPolicy[NodeEvalT]
    proof_policy: ProofPolicy[NodeEvalT]

    def __init__(
        self,
        *,
        aggregation_policy: AggregationPolicy[NodeEvalT] | None = None,
        proof_policy: ProofPolicy[NodeEvalT],
    ) -> None:
        """Initialize the shared backup engine with injected aggregation/proof policies."""
        if aggregation_policy is None:
            aggregation_policy = BestChildAggregationPolicy()
        self.aggregation_policy = aggregation_policy
        self.proof_policy = proof_policy

    def backup_from_children(
        self,
        node_eval: NodeEvalT,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Recompute ``backed_up_value`` and PV from updated child values."""
        best_branch_before_update = node_eval.best_branch()
        self.prepare_aggregation(
            node_eval=node_eval,
            branches_with_updated_value=branches_with_updated_value,
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
            update_pv=lambda: self._update_pv_generic(
                node_eval=node_eval,
                best_branch_before_update=best_branch_before_update,
                selection=selection,
                branches_with_updated_best_branch_seq=(
                    branches_with_updated_best_branch_seq
                ),
            ),
        )

    def prepare_aggregation(
        self,
        *,
        node_eval: NodeEvalT,
        branches_with_updated_value: set[BranchKey],
    ) -> None:
        """Refresh child ordering before aggregation when child values changed."""
        if not branches_with_updated_value:
            return

        prepare_best_child_aggregation(
            node_eval=node_eval,
            branches_to_consider=branches_with_updated_value,
        )

    def _update_pv_generic(
        self,
        *,
        node_eval: NodeEvalT,
        best_branch_before_update: BranchKey | None,
        selection: SelectedValue,
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> bool:
        """Refresh the principal variation using the shared best-child algorithm."""
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
        node_eval: NodeEvalT,
        best_branch_before_update: BranchKey | None,
        best_branch_after_update: BranchKey,
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> bool:
        """Return whether the PV head should be rebuilt from the current best child."""
        return (
            best_branch_before_update != best_branch_after_update
            or not node_eval.best_branch_sequence
            or node_eval.best_branch_sequence[0] != best_branch_after_update
            or best_branch_after_update in branches_with_updated_best_branch_seq
        )
