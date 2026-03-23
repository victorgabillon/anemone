"""Public construction API for Anemone search runtimes.

Preferred entrypoints:

* ``SearchArgs`` for build-time configuration
* ``create_search(...)`` for the runnable runtime
* ``SearchRuntime`` for the runtime object

Internal collaborator wiring lives in ``anemone._runtime_assembly``. Legacy
``TreeAndValue...`` names remain available here for compatibility.
"""
# pylint: disable=duplicate-code

from collections.abc import Hashable
from dataclasses import dataclass
from random import Random
from typing import Literal

from valanga import Dynamics, RepresentationFactory, StateModifications
from valanga.evaluator_types import EvaluatorInput
from valanga.policy import NotifyProgressCallable

from anemone import node_selector as node_selector_m
from anemone import recommender_rule
from anemone._runtime_assembly import (
    SearchRuntimeDependencies,
    assemble_search_runtime_dependencies,
)
from anemone._valanga_types import AnyTurnState
from anemone.dynamics import SearchDynamics
from anemone.hooks.search_hooks import SearchHooks
from anemone.indices.node_indices.index_types import IndexComputationType
from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
)
from anemone.node_evaluation.tree.factory import (
    NodeTreeEvaluationFactory,
    NodeTreeMinmaxEvaluationFactory,
)
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
)
from anemone.tree_and_value_branch_selector import (
    SearchRecommender,
    TreeAndValueBranchSelector,
)
from anemone.tree_exploration import (
    SearchRuntime,
    TreeExploration,
    create_tree_exploration,
)

TREE_AND_VALUE_LITERAL_STRING: Literal["TreeAndValue"] = "TreeAndValue"


# Public config aliases
@dataclass
class TreeAndValuePlayerArgs:
    """Configuration for building one search runtime.

    ``SearchArgs`` is the preferred public alias. This legacy name remains for
    compatibility with older imports and configuration code.
    """

    node_selector: ComposedNodeSelectorArgs
    opening_type: node_selector_m.OpeningType
    stopping_criterion: AllStoppingCriterionArgs
    recommender_rule: recommender_rule.AllRecommendFunctionsArgs
    index_computation: IndexComputationType | None = None
    type: Literal["TreeAndValue"] = TREE_AND_VALUE_LITERAL_STRING


SearchArgs = TreeAndValuePlayerArgs

# Public exports
__all__ = [
    "SearchArgs",
    "SearchRecommender",
    "SearchRuntime",
    "TreeAndValueBranchSelector",
    "TreeAndValuePlayerArgs",
    "create_search",
    "create_search_with_tree_eval_factory",
    "create_tree_and_value_branch_selector",
    "create_tree_and_value_branch_selector_with_tree_eval_factory",
    "create_tree_and_value_exploration",
    "create_tree_and_value_exploration_with_tree_eval_factory",
]


# Preferred runtime constructors
def create_search[StateT: AnyTurnState, ActionT: Hashable](
    state_type: type[StateT],
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    starting_state: StateT,
    args: SearchArgs,
    random_generator: Random,
    master_state_value_evaluator: MasterStateValueEvaluator,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None,
    notify_progress: NotifyProgressCallable | None = None,
    hooks: SearchHooks | None = None,
) -> TreeExploration[AlgorithmNode[StateT]]:
    """Build the preferred runnable ``TreeExploration`` runtime."""
    _ = state_type

    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT]
    node_tree_evaluation_factory = NodeTreeMinmaxEvaluationFactory[StateT]()

    return create_search_with_tree_eval_factory(
        state_type=state_type,
        dynamics=dynamics,
        starting_state=starting_state,
        args=args,
        random_generator=random_generator,
        master_state_evaluator=master_state_value_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=node_tree_evaluation_factory,
        notify_progress=notify_progress,
        hooks=hooks,
    )


def create_search_with_tree_eval_factory[
    StateT: AnyTurnState,
    ActionT: Hashable,
](
    state_type: type[StateT],
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    starting_state: StateT,
    args: SearchArgs,
    random_generator: Random,
    master_state_evaluator: MasterStateValueEvaluator,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None,
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT],
    notify_progress: NotifyProgressCallable | None = None,
    hooks: SearchHooks | None = None,
) -> TreeExploration[AlgorithmNode[StateT]]:
    """Build the runtime with an explicit tree-evaluation family."""
    _ = state_type

    dependencies = assemble_search_runtime_dependencies(
        dynamics=dynamics,
        args=args,
        random_generator=random_generator,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=node_tree_evaluation_factory,
        hooks=hooks,
    )

    return _build_runtime_from_dependencies(
        starting_state=starting_state,
        args=args,
        dependencies=dependencies,
        notify_progress=notify_progress,
    )


# Recommendation-only wrapper constructors
def create_tree_and_value_branch_selector[StateT: AnyTurnState, ActionT: Hashable](
    state_type: type[StateT],
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    args: TreeAndValuePlayerArgs,
    random_generator: Random,
    master_state_value_evaluator: MasterStateValueEvaluator,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None,
    hooks: SearchHooks | None = None,
) -> TreeAndValueBranchSelector[StateT]:
    """Build the secondary recommend-only wrapper over the runtime."""
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT]
    node_tree_evaluation_factory = NodeTreeMinmaxEvaluationFactory[StateT]()

    return create_tree_and_value_branch_selector_with_tree_eval_factory(
        state_type=state_type,
        dynamics=dynamics,
        args=args,
        random_generator=random_generator,
        master_state_evaluator=master_state_value_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=node_tree_evaluation_factory,
        hooks=hooks,
    )


def create_tree_and_value_branch_selector_with_tree_eval_factory[
    StateT: AnyTurnState,
    ActionT: Hashable,
](
    state_type: type[StateT],
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    args: TreeAndValuePlayerArgs,
    random_generator: Random,
    master_state_evaluator: MasterStateValueEvaluator,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None,
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT],
    hooks: SearchHooks | None = None,
) -> TreeAndValueBranchSelector[StateT]:
    """Build the recommend-only wrapper with an explicit value family."""
    _ = state_type

    dependencies = assemble_search_runtime_dependencies(
        dynamics=dynamics,
        args=args,
        random_generator=random_generator,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=node_tree_evaluation_factory,
        hooks=hooks,
    )

    return _build_recommender_from_dependencies(
        args=args,
        random_generator=random_generator,
        dependencies=dependencies,
    )


# Compatibility wrappers
def create_tree_and_value_exploration[StateT: AnyTurnState, ActionT: Hashable](
    state_type: type[StateT],
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    starting_state: StateT,
    args: TreeAndValuePlayerArgs,
    random_generator: Random,
    master_state_value_evaluator: MasterStateValueEvaluator,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None,
    notify_progress: NotifyProgressCallable | None = None,
    hooks: SearchHooks | None = None,
) -> SearchRuntime[AlgorithmNode[StateT]]:
    """Compatibility wrapper over ``create_search(...)``."""
    return create_search(
        state_type=state_type,
        dynamics=dynamics,
        starting_state=starting_state,
        args=args,
        random_generator=random_generator,
        master_state_value_evaluator=master_state_value_evaluator,
        state_representation_factory=state_representation_factory,
        notify_progress=notify_progress,
        hooks=hooks,
    )


def create_tree_and_value_exploration_with_tree_eval_factory[
    StateT: AnyTurnState,
    ActionT: Hashable,
](
    state_type: type[StateT],
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    starting_state: StateT,
    args: TreeAndValuePlayerArgs,
    random_generator: Random,
    master_state_evaluator: MasterStateValueEvaluator,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None,
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT],
    notify_progress: NotifyProgressCallable | None = None,
    hooks: SearchHooks | None = None,
) -> SearchRuntime[AlgorithmNode[StateT]]:
    """Compatibility wrapper over ``create_search_with_tree_eval_factory(...)``."""
    return create_search_with_tree_eval_factory(
        state_type=state_type,
        dynamics=dynamics,
        starting_state=starting_state,
        args=args,
        random_generator=random_generator,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=node_tree_evaluation_factory,
        notify_progress=notify_progress,
        hooks=hooks,
    )


# Local build helpers
def _build_runtime_from_dependencies[StateT: AnyTurnState](
    *,
    starting_state: StateT,
    args: SearchArgs,
    dependencies: SearchRuntimeDependencies[StateT],
    notify_progress: NotifyProgressCallable | None = None,
) -> SearchRuntime[AlgorithmNode[StateT]]:
    """Assemble the runtime from already-wired collaborators."""
    return create_tree_exploration(
        node_selector_create=dependencies.node_selector_create,
        starting_state=starting_state,
        tree_manager=dependencies.tree_manager,
        tree_factory=dependencies.tree_factory,
        stopping_criterion_args=args.stopping_criterion,
        recommend_branch_after_exploration=args.recommender_rule,
        notify_percent_function=notify_progress,
    )


def _build_recommender_from_dependencies[StateT: AnyTurnState](
    *,
    args: SearchArgs,
    random_generator: Random,
    dependencies: SearchRuntimeDependencies[StateT],
) -> SearchRecommender[StateT]:
    """Assemble the secondary recommend-only wrapper from shared wiring."""
    return TreeAndValueBranchSelector(
        tree_manager=dependencies.tree_manager,
        random_generator=random_generator,
        tree_factory=dependencies.tree_factory,
        node_selector_create=dependencies.node_selector_create,
        stopping_criterion_args=args.stopping_criterion,
        recommend_branch_after_exploration=args.recommender_rule,
    )
