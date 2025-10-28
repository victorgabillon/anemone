"""
MoveAndValueTreeFactory
"""

from typing import TYPE_CHECKING

from valanga import State

import anemone.node_factory as nod_fac
from anemone.node_evaluator import (
    EvaluationQueries,
    NodeEvaluator,
)
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from .descendants import RangedDescendants
from .value_tree import ValueTree

if TYPE_CHECKING:
    from anemone.nodes.itree_node import ITreeNode


class ValueTreeFactory:
    """
    MoveAndValueTreeFactory
    """

    node_factory: nod_fac.AlgorithmNodeFactory
    node_evaluator: NodeEvaluator

    def __init__(
        self, node_factory: nod_fac.AlgorithmNodeFactory, node_evaluator: NodeEvaluator
    ) -> None:
        """
        creates the tree factory
        Args:
            node_factory:
            node_evaluator:
        """
        self.node_factory = node_factory
        self.node_evaluator = node_evaluator

    def create(self, starting_state: State) -> ValueTree:
        """
        creates the tree

        Args:
            starting_state: the starting position

        Returns:

        """

        root_node: ITreeNode = self.node_factory.create(
            state=starting_state,
            tree_depth=0,  # by default
            count=0,
            parent_node=None,
            modifications=None,
            branch_from_parent=None,
        )

        evaluation_queries: EvaluationQueries = EvaluationQueries()

        assert isinstance(root_node, AlgorithmNode)
        self.node_evaluator.add_evaluation_query(
            node=root_node, evaluation_queries=evaluation_queries
        )

        self.node_evaluator.evaluate_all_queried_nodes(
            evaluation_queries=evaluation_queries
        )
        # is this needed? used outside?

        descendants: RangedDescendants = RangedDescendants()
        descendants.add_descendant(root_node)

        value_tree: ValueTree = ValueTree(root_node=root_node, descendants=descendants)

        return value_tree
