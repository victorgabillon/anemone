"""Provide factories for node tree evaluation implementations."""

from typing import Any, Protocol

from valanga import State, TurnState

from anemone.backup_policies import ExplicitMinimaxBackupPolicy
from anemone.backup_policies.protocols import BackupPolicy
from anemone.node_evaluation.node_tree_evaluation.node_adversarial_evaluation import (
    NodeAdversarialEvaluation,
)
from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import AdversarialZeroSumObjective, Objective


class NodeTreeMinmaxEvaluationFactory[StateT: TurnState]:
    """The class creating Node Evaluations including children."""

    def __init__(
        self,
        backup_policy: BackupPolicy[NodeMinmaxEvaluation[Any, StateT]] | None = None,
        objective: Objective[StateT] | None = None,
    ) -> None:
        """Initialize the factory with an explicit backup policy."""
        if backup_policy is not None:
            self.backup_policy = backup_policy
        else:
            self.backup_policy = ExplicitMinimaxBackupPolicy()
        if objective is not None:
            self.objective = objective
        else:
            self.objective = AdversarialZeroSumObjective()

    def create(
        self,
        tree_node: TreeNode[Any, StateT],
    ) -> NodeMinmaxEvaluation[Any, StateT]:
        """Create a new NodeEvaluationIncludingChildren object.

        Args:
            tree_node (TreeNode): The tree node for which the evaluation is created.

        Returns:
            NodeEvaluationIncludingChildren: The newly created NodeEvaluationIncludingChildren object.

        """
        return NodeMinmaxEvaluation(
            tree_node=tree_node,
            backup_policy=self.backup_policy,
            objective=self.objective,
        )


class NodeTreeEvaluationFactory[T: State = State](Protocol):
    """The class creating adversarial node evaluations including children."""

    def create(
        self,
        tree_node: TreeNode[Any, T],
    ) -> NodeAdversarialEvaluation[T]:
        """Create an adversarial node evaluation instance for the given node."""
        ...
