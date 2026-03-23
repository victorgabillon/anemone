"""Thin minimax wrapper that installs adversarial defaults on the shared engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.backup_policies.aggregation import (
    AggregationPolicy,
    BestChildAggregationPolicy,
)
from anemone.backup_policies.explicit_tree import ExplicitTreeBackupPolicy
from anemone.backup_policies.proof import MinimaxProofPolicy, ProofPolicy

if TYPE_CHECKING:
    from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
        NodeMinmaxEvaluation,
    )


class ExplicitMinimaxBackupPolicy(
    ExplicitTreeBackupPolicy["NodeMinmaxEvaluation[Any, Any]"]
):
    """Named minimax wrapper around ``ExplicitTreeBackupPolicy``."""

    def __init__(
        self,
        aggregation_policy: AggregationPolicy[NodeMinmaxEvaluation[Any, Any]]
        | None = None,
        proof_policy: ProofPolicy[NodeMinmaxEvaluation[Any, Any]] | None = None,
    ) -> None:
        """Install the minimax family's default aggregation and proof policies."""
        if aggregation_policy is None:
            aggregation_policy = BestChildAggregationPolicy()
        if proof_policy is None:
            proof_policy = MinimaxProofPolicy()
        super().__init__(
            aggregation_policy=aggregation_policy,
            proof_policy=proof_policy,
        )
