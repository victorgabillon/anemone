"""Provide a factory function for creating an AlgorithmNodeTreeManager object."""

from typing import TYPE_CHECKING, Any

from anemone.dynamics import SearchDynamics
from anemone.indices.index_manager import (
    NodeExplorationIndexManager,
    create_exploration_index_manager,
)
from anemone.indices.node_indices.index_types import (
    IndexComputationType,
)
from anemone.node_evaluation.direct import (
    EvaluationQueries,
    NodeDirectEvaluator,
)
from anemone.node_factory import (
    AlgorithmNodeFactory,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.updates.depth_index_propagator import DepthIndexPropagator

from .algorithm_node_tree_manager import AlgorithmNodeTreeManager
from .opening_expansion_config import OpeningExpansionConfig
from .tree_manager import TreeManager

if TYPE_CHECKING:
    from anemone.rollouts import RolloutActionSelector


def create_algorithm_node_tree_manager(
    node_direct_evaluator: NodeDirectEvaluator[Any] | None,
    algorithm_node_factory: AlgorithmNodeFactory[Any],
    dynamics: SearchDynamics[Any, Any],
    index_computation: IndexComputationType | None,
    depth_index: bool = False,
    opening_expansion_config: OpeningExpansionConfig | None = None,
    rollout_action_selector: "RolloutActionSelector[AlgorithmNode] | None" = None,
) -> AlgorithmNodeTreeManager:
    """Create an AlgorithmNodeTreeManager object.

    Args:
        node_direct_evaluator: The NodeDirectEvaluator used for evaluating nodes in the tree.
        algorithm_node_factory: The AlgorithmNodeFactory object used for creating algorithm nodes.
        index_computation: The type of index computation to be used.
        dynamics: The SearchDynamics object used for labeling the edges in the visualization.
        depth_index: Whether to maintain descendant-depth metadata incrementally.
        opening_expansion_config: Optional opening expansion strategy config.
        rollout_action_selector: Optional runtime rollout action selector override.

    Returns:
        An AlgorithmNodeTreeManager object.

    """
    tree_manager: TreeManager[AlgorithmNode] = TreeManager[AlgorithmNode](
        node_factory=algorithm_node_factory,
        dynamics=dynamics,
    )

    evaluation_queries: EvaluationQueries = EvaluationQueries()

    exploration_index_manager: NodeExplorationIndexManager = (
        create_exploration_index_manager(index_computation=index_computation)
    )

    algorithm_node_tree_manager: AlgorithmNodeTreeManager = AlgorithmNodeTreeManager(
        node_evaluator=node_direct_evaluator,
        tree_manager=tree_manager,
        evaluation_queries=evaluation_queries,
        index_manager=exploration_index_manager,
        depth_index_propagator=(DepthIndexPropagator() if depth_index else None),
        opening_expansion_config=opening_expansion_config or OpeningExpansionConfig(),
        rollout_action_selector=rollout_action_selector,
    )

    return algorithm_node_tree_manager
