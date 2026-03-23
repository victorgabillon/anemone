"""Thin max wrapper that installs single-agent defaults on the shared engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.backup_policies.aggregation import (
    AggregationPolicy,
    BestChildAggregationPolicy,
)
from anemone.backup_policies.explicit_tree import ExplicitTreeBackupPolicy
from anemone.backup_policies.proof import MaxProofPolicy, ProofPolicy

if TYPE_CHECKING:
    from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
        NodeMaxEvaluation,
    )


class ExplicitMaxBackupPolicy(ExplicitTreeBackupPolicy["NodeMaxEvaluation[Any]"]):
    """Named single-agent max wrapper around ``ExplicitTreeBackupPolicy``."""

    def __init__(
        self,
        aggregation_policy: AggregationPolicy[NodeMaxEvaluation[Any]] | None = None,
        proof_policy: ProofPolicy[NodeMaxEvaluation[Any]] | None = None,
    ) -> None:
        """Install the max family's default aggregation and proof policies."""
        if aggregation_policy is None:
            aggregation_policy = BestChildAggregationPolicy()
        if proof_policy is None:
            proof_policy = MaxProofPolicy()
        super().__init__(
            aggregation_policy=aggregation_policy,
            proof_policy=proof_policy,
        )
