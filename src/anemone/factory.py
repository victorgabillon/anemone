"""
This module provides functions for creating a TreeAndValueMoveSelector object.

The TreeAndValueMoveSelector is a player that uses a tree-based approach to select moves in a game. It evaluates the
game tree using a node evaluator and selects moves based on a set of criteria defined by the node selector. The player
uses a stopping criterion to determine when to stop the search and a recommender rule to recommend a move after
exploration.

This module also provides functions for creating the necessary components of the TreeAndValueMoveSelector, such as the
node evaluator, node selector, tree factory, and tree manager.

"""

import queue
import random
from dataclasses import dataclass
from typing import Literal, Type

from valanga import ContentRepresentation, RepresentationFactory, TurnState

import anemone.search_factory as search_factories
from anemone import node_factory
from anemone.node_evaluation.node_direct_evaluation.factory import create_node_evaluator
from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
    NodeDirectEvaluator,
)
from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation_factory import (
    NodeTreeEvaluationFactory,
    NodeTreeMinmaxEvaluationFactory,
)
from anemone.node_factory.base import TreeNodeFactory
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
)
from anemone.utils.dataclass import IsDataclass

from . import node_selector as node_selector_m
from . import recommender_rule
from . import tree_manager as tree_man
from .indices.node_indices.index_types import IndexComputationType
from .tree_and_value_branch_selector import TreeAndValueBranchSelector
from .trees.factory import ValueTreeFactory

TreeAndValueLiteralString: Literal["TreeAndValue"] = "TreeAndValue"


@dataclass
class TreeAndValuePlayerArgs:
    """
    Data class for the arguments of a TreeAndValueMoveSelector.
    """

    node_selector: node_selector_m.AllNodeSelectorArgs
    opening_type: node_selector_m.OpeningType
    stopping_criterion: AllStoppingCriterionArgs
    recommender_rule: recommender_rule.AllRecommendFunctionsArgs
    index_computation: IndexComputationType | None = None
    type: Literal["TreeAndValue"] = TreeAndValueLiteralString


def create_tree_and_value_branch_selector[StateT: TurnState](
    state_type: Type[StateT],
    args: TreeAndValuePlayerArgs,
    random_generator: random.Random,
    master_state_evaluator: MasterStateEvaluator,
    state_representation_factory: RepresentationFactory[ContentRepresentation] | None,
    queue_progress_player: queue.Queue[IsDataclass] | None,
) -> TreeAndValueBranchSelector[StateT]:
    """Convenience constructor using the default minmax tree evaluation.

    This keeps the existing API stable, while allowing advanced users to inject a
    different tree-evaluation strategy via
    `create_tree_and_value_branch_selector_with_tree_eval_factory`.
    """

    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT]
    node_tree_evaluation_factory = NodeTreeMinmaxEvaluationFactory[StateT]()

    return create_tree_and_value_branch_selector_with_tree_eval_factory(
        state_type=state_type,
        args=args,
        random_generator=random_generator,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=node_tree_evaluation_factory,
        queue_progress_player=queue_progress_player,
    )


def create_tree_and_value_branch_selector_with_tree_eval_factory[StateT: TurnState](
    state_type: Type[StateT],
    args: TreeAndValuePlayerArgs,
    random_generator: random.Random,
    master_state_evaluator: MasterStateEvaluator,
    state_representation_factory: RepresentationFactory[ContentRepresentation] | None,
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT],
    queue_progress_player: queue.Queue[IsDataclass] | None,
) -> TreeAndValueBranchSelector[StateT]:
    """
    Create a TreeAndValueBranchSelector object with the given arguments.

    Args:
        args (TreeAndValuePlayerArgs): The arguments for creating the TreeAndValueBranchSelector.
        syzygy (SyzygyTable | None): The SyzygyTable object for tablebase endgame evaluation.
        random_generator (random.Random): The random number generator.

    Returns:
        TreeAndValueBranchSelector: The created TreeAndValueBranchSelector object.

    """

    node_evaluator: NodeDirectEvaluator[StateT] = create_node_evaluator(
        master_state_evaluator=master_state_evaluator,
    )

    tree_node_factory: TreeNodeFactory[AlgorithmNode[StateT], StateT] = TreeNodeFactory[
        AlgorithmNode[StateT], StateT
    ]()

    search_factory: search_factories.SearchFactoryP = search_factories.SearchFactory(
        node_selector_args=args.node_selector,
        opening_type=args.opening_type,
        random_generator=random_generator,
        index_computation=args.index_computation,
    )

    algorithm_node_factory: node_factory.AlgorithmNodeFactory[StateT]
    algorithm_node_factory = node_factory.AlgorithmNodeFactory[StateT](
        tree_node_factory=tree_node_factory,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=node_tree_evaluation_factory,
        exploration_index_data_create=search_factory.node_index_create,
    )

    tree_factory: ValueTreeFactory[StateT] = ValueTreeFactory[StateT](
        node_factory=algorithm_node_factory,
        node_direct_evaluator=node_evaluator,
    )

    tree_manager: tree_man.AlgorithmNodeTreeManager
    tree_manager = tree_man.create_algorithm_node_tree_manager(
        algorithm_node_factory=algorithm_node_factory,
        node_direct_evaluator=node_evaluator,
        index_computation=args.index_computation,
        index_updater=search_factory.create_node_index_updater(),
    )

    tree_move_selector: TreeAndValueBranchSelector[StateT] = TreeAndValueBranchSelector(
        tree_manager=tree_manager,
        random_generator=random_generator,
        tree_factory=tree_factory,
        node_selector_create=search_factory.create_node_selector_factory(),
        stopping_criterion_args=args.stopping_criterion,
        recommend_move_after_exploration=args.recommender_rule,
        queue_progress_player=queue_progress_player,
    )
    return tree_move_selector
