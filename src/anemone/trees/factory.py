"""
ValueTreeFactory
"""

from valanga import State

import anemone.node_factory as nod_fac
from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    EvaluationQueries,
    NodeDirectEvaluator,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.trees.tree import Tree
from .descendants import RangedDescendants


class ValueTreeFactory[TState: State = State]:
    """
    ValueTreeFactory
    """

    node_factory: nod_fac.AlgorithmNodeFactory[TState]
    node_direct_evaluator: NodeDirectEvaluator[TState]

    def __init__(
        self,
        node_factory: nod_fac.AlgorithmNodeFactory[TState],
        node_direct_evaluator: NodeDirectEvaluator[TState],
    ) -> None:
        """
        creates the tree factory
        Args:
            node_factory:
            node_evaluator:
        """
        self.node_factory = node_factory
        self.node_direct_evaluator = node_direct_evaluator

    def create(self, starting_state: TState) -> Tree[AlgorithmNode[TState]]:
        """
        creates the tree

        Args:
            starting_state: the starting position

        Returns:

        """

        root_node: AlgorithmNode[TState] = self.node_factory.create(
            state=starting_state,
            tree_depth=0,  # by default
            count=0,
            parent_node=None,
            modifications=None,
            branch_from_parent=None,
        )

        evaluation_queries: EvaluationQueries[TState] = EvaluationQueries()

        self.node_direct_evaluator.add_evaluation_query(
            node=root_node, evaluation_queries=evaluation_queries
        )

        self.node_direct_evaluator.evaluate_all_queried_nodes(
            evaluation_queries=evaluation_queries
        )
        # is this needed? used outside?

        descendants: RangedDescendants[AlgorithmNode[TState]] = RangedDescendants[AlgorithmNode[TState]]()
        descendants.add_descendant(root_node)

        value_tree: Tree[AlgorithmNode[TState]] = Tree[AlgorithmNode[TState]](
            root_node=root_node, descendants=descendants
        )

        return value_tree
