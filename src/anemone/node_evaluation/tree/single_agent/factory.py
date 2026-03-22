"""Provide the single-agent max factory built on shared tree-evaluation wiring."""

from anemone._valanga_types import AnyTurnState
from anemone.backup_policies.protocols import BackupPolicy
from anemone.node_evaluation.tree.factory import (
    ConfiguredNodeTreeEvaluationFactory,
    resolve_factory_dependency,
)
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
    make_default_backup_policy,
    make_default_objective,
)
from anemone.objectives import Objective


class NodeMaxEvaluationFactory[StateT: AnyTurnState = AnyTurnState](
    ConfiguredNodeTreeEvaluationFactory[StateT, NodeMaxEvaluation[StateT]]
):
    """Create single-agent max evaluations via the shared tree-evaluation assembly."""

    def __init__(
        self,
        backup_policy: BackupPolicy[NodeMaxEvaluation[StateT]] | None = None,
        objective: Objective[StateT] | None = None,
    ) -> None:
        """Initialize the factory with optional explicit objective and backup policy."""
        super().__init__(
            evaluation_type=NodeMaxEvaluation,
            backup_policy=resolve_factory_dependency(
                backup_policy,
                default_factory=make_default_backup_policy,
            ),
            objective=resolve_factory_dependency(
                objective,
                default_factory=make_default_objective,
            ),
        )
