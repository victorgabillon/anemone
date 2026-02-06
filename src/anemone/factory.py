"""Module for creating Tree and Value Branch Selector objects."""

from dataclasses import dataclass
from random import Random
from typing import Literal

from valanga import RepresentationFactory, StateModifications, TurnState
from valanga.evaluator_types import EvaluatorInput

from anemone import node_factory
from anemone import search_factory as search_factories
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

from . import node_selector as node_selector_m
from . import recommender_rule
from . import tree_manager as tree_man
from .indices.node_indices.index_types import IndexComputationType
from .tree_and_value_branch_selector import TreeAndValueBranchSelector
from .trees.factory import ValueTreeFactory

TREE_AND_VALUE_LITERAL_STRING: Literal["TreeAndValue"] = "TreeAndValue"


@dataclass
class TreeAndValuePlayerArgs:
    """Dataclass for Tree and Value Player Arguments."""

    node_selector: node_selector_m.AllNodeSelectorArgs
    opening_type: node_selector_m.OpeningType
    stopping_criterion: AllStoppingCriterionArgs
    recommender_rule: recommender_rule.AllRecommendFunctionsArgs
    index_computation: IndexComputationType | None = None
    type: Literal["TreeAndValue"] = TREE_AND_VALUE_LITERAL_STRING


def create_tree_and_value_branch_selector[StateT: TurnState](
    state_type: type[StateT],
    args: TreeAndValuePlayerArgs,
    random_generator: Random,
    master_state_evaluator: MasterStateEvaluator,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None,
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
    )


def create_tree_and_value_branch_selector_with_tree_eval_factory[StateT: TurnState](
    state_type: type[StateT],
    args: TreeAndValuePlayerArgs,
    random_generator: Random,
    master_state_evaluator: MasterStateEvaluator,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None,
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT],
) -> TreeAndValueBranchSelector[StateT]:
    """Create a TreeAndValueBranchSelector object with the given arguments.

    Args:
        args (TreeAndValuePlayerArgs): The arguments for creating the TreeAndValueBranchSelector.
        syzygy (SyzygyTable | None): The SyzygyTable object for tablebase endgame evaluation.
        random_generator (random.Random): The random number generator.

    Returns:
        TreeAndValueBranchSelector: The created TreeAndValueBranchSelector object.

    """
    _ = state_type  # not used here

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

    tree_branch_selector: TreeAndValueBranchSelector[StateT] = (
        TreeAndValueBranchSelector(
            tree_manager=tree_manager,
            random_generator=random_generator,
            tree_factory=tree_factory,
            node_selector_create=search_factory.create_node_selector_factory(),
            stopping_criterion_args=args.stopping_criterion,
            recommend_branch_after_exploration=args.recommender_rule,
        )
    )
    return tree_branch_selector
