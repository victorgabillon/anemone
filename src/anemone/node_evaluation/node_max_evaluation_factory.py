"""Provide a minimal factory for the single-agent max node-evaluation family."""

from typing import Any

from valanga import State

from anemone.backup_policies import ExplicitMaxBackupPolicy
from anemone.backup_policies.protocols import BackupPolicy
from anemone.node_evaluation.node_max_evaluation import NodeMaxEvaluation
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import Objective, SingleAgentMaxObjective


class NodeMaxEvaluationFactory[StateT: State = State]:
    """Create single-agent max node evaluations with explicit default dependencies."""

    def __init__(
        self,
        backup_policy: BackupPolicy[NodeMaxEvaluation[Any]] | None = None,
        objective: Objective[StateT] | None = None,
    ) -> None:
        """Initialize the factory with optional explicit objective and backup policy."""
        if backup_policy is not None:
            self.backup_policy = backup_policy
        else:
            self.backup_policy = ExplicitMaxBackupPolicy()
        if objective is not None:
            self.objective = objective
        else:
            self.objective = SingleAgentMaxObjective()

    def create(
        self,
        tree_node: TreeNode[Any, StateT],
    ) -> NodeMaxEvaluation[StateT]:
        """Create a single-agent max node evaluation for the given tree node."""
        return NodeMaxEvaluation(
            tree_node=tree_node,
            backup_policy=self.backup_policy,
            objective=self.objective,
        )
