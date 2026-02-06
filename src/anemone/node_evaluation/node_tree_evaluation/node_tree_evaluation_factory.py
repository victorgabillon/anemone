"""Provide factories for node tree evaluation implementations."""


from typing import Any, Protocol

from valanga import State, TurnState

from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation import (
    NodeTreeEvaluation,
)
from anemone.nodes.tree_node import TreeNode


class NodeTreeMinmaxEvaluationFactory[StateT: TurnState]:
    """The class creating Node Evaluations including children."""

    def create(
        self,
        tree_node: TreeNode[Any, StateT],
    ) -> NodeMinmaxEvaluation[Any, StateT]:
        """Creates a new NodeEvaluationIncludingChildren object.

        Args:
            tree_node (TreeNode): The tree node for which the evaluation is created.

        Returns:
            NodeEvaluationIncludingChildren: The newly created NodeEvaluationIncludingChildren object.

        """
        return NodeMinmaxEvaluation(tree_node=tree_node)


class NodeTreeEvaluationFactory[StateT2: State = State](Protocol):
    """The class creating Node Evaluations including children."""

    def create(
        self,
        tree_node: TreeNode[Any, StateT2],
    ) -> NodeTreeEvaluation[StateT2]:
        """Create a NodeTreeEvaluation instance for the given node."""
        ...
