"""Provide a factory function for creating an AlgorithmNodeTreeManager object.

The AlgorithmNodeTreeManager is responsible for managing the tree structure of algorithm nodes,
performing updates on the nodes, and handling evaluation queries.

"""

from typing import Any

from anemone import updates as upda
from anemone.indices.index_manager import (
    NodeExplorationIndexManager,
    create_exploration_index_manager,
)
from anemone.indices.node_indices.index_types import (
    IndexComputationType,
)
from anemone.node_evaluation.node_direct_evaluation import (
    EvaluationQueries,
    NodeDirectEvaluator,
)
from anemone.node_factory import (
    AlgorithmNodeFactory,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.state_transition import ValangaStateTransition
from anemone.updates.index_updater import IndexUpdater

from .algorithm_node_tree_manager import AlgorithmNodeTreeManager
from .tree_manager import TreeManager


def create_algorithm_node_tree_manager(
    node_direct_evaluator: NodeDirectEvaluator[Any] | None,
    algorithm_node_factory: AlgorithmNodeFactory[Any],
    index_computation: IndexComputationType | None,
    index_updater: IndexUpdater | None,
) -> AlgorithmNodeTreeManager:
    """Create an AlgorithmNodeTreeManager object.

    Args:
        node_direct_evaluator: The NodeDirectEvaluator used for evaluating nodes in the tree.
        algorithm_node_factory: The AlgorithmNodeFactory object used for creating algorithm nodes.
        index_computation: The type of index computation to be used.
        index_updater: The IndexUpdater object used for updating the indices.

    Returns:
        An AlgorithmNodeTreeManager object.

    """
    tree_manager: TreeManager[AlgorithmNode] = TreeManager[AlgorithmNode](
        node_factory=algorithm_node_factory,
        transition=ValangaStateTransition(),
    )

    algorithm_node_updater: upda.AlgorithmNodeUpdater = (
        upda.create_algorithm_node_updater(index_updater=index_updater)
    )

    evaluation_queries: EvaluationQueries = EvaluationQueries()

    exploration_index_manager: NodeExplorationIndexManager = (
        create_exploration_index_manager(index_computation=index_computation)
    )

    algorithm_node_tree_manager: AlgorithmNodeTreeManager = AlgorithmNodeTreeManager(
        node_evaluator=node_direct_evaluator,
        tree_manager=tree_manager,
        algorithm_node_updater=algorithm_node_updater,
        algorithm_tree_node_factory=algorithm_node_factory,
        evaluation_queries=evaluation_queries,
        index_manager=exploration_index_manager,
    )

    return algorithm_node_tree_manager
