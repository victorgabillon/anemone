"""Top-level assembly helpers for Anemone search runtimes.

Preferred public vocabulary:

* ``SearchArgs`` for build-time configuration
* ``create_search(...)`` for building a runnable runtime
* ``SearchRuntime`` for the runtime object itself

Legacy ``TreeAndValue...`` names remain available for compatibility.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Literal

from valanga import Dynamics, RepresentationFactory, StateModifications
from valanga.evaluator_types import EvaluatorInput
from valanga.policy import NotifyProgressCallable

from anemone import node_factory
from anemone import search_factory as search_factories
from anemone._valanga_types import AnyTurnState
from anemone.dynamics import SearchDynamics, normalize_search_dynamics
from anemone.hooks.search_hooks import SearchHooks
from anemone.node_evaluation.direct.factory import create_node_evaluator
from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
)
from anemone.node_evaluation.tree.factory import (
    NodeTreeEvaluationFactory,
    NodeTreeMinmaxEvaluationFactory,
)
from anemone.node_factory.base import TreeNodeFactory
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
)

from . import node_selector as node_selector_m
from . import recommender_rule
from . import tree_manager as tree_man
from .indices.node_indices.index_types import IndexComputationType
from .tree_and_value_branch_selector import TreeAndValueBranchSelector
from .tree_exploration import SearchRuntime, TreeExploration, create_tree_exploration
from .trees.factory import ValueTreeFactory

if TYPE_CHECKING:
    from anemone.node_evaluation.direct.node_direct_evaluator import (
        NodeDirectEvaluator,
    )

TREE_AND_VALUE_LITERAL_STRING: Literal["TreeAndValue"] = "TreeAndValue"


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


@dataclass(frozen=True, slots=True)
class _TreeAndValueRuntimeDependencies[StateT: AnyTurnState]:
    """Reusable collaborators needed to assemble one ``TreeExploration``."""

    tree_manager: tree_man.AlgorithmNodeTreeManager[AlgorithmNode[StateT]]
    tree_factory: ValueTreeFactory[StateT]
    node_selector_create: search_factories.NodeSelectorFactory


def _assemble_tree_and_value_runtime_dependencies[
    StateT: AnyTurnState,
    ActionT: Hashable,
](
    *,
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
) -> _TreeAndValueRuntimeDependencies[StateT]:
    """Assemble the reusable collaborators behind one tree-exploration runtime."""
    search_dynamics = normalize_search_dynamics(dynamics)

    node_evaluator: NodeDirectEvaluator[StateT] = create_node_evaluator(
        master_state_evaluator=master_state_evaluator,
    )

    tree_node_factory: TreeNodeFactory[AlgorithmNode[StateT], StateT] = TreeNodeFactory[
        AlgorithmNode[StateT], StateT
    ]()

    search_factory = search_factories.SearchFactory(
        node_selector_args=args.node_selector,
        opening_type=args.opening_type,
        random_generator=random_generator,
        dynamics=search_dynamics,
        index_computation=args.index_computation,
        hooks=hooks,
    )

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

    tree_manager = tree_man.create_algorithm_node_tree_manager(
        algorithm_node_factory=algorithm_node_factory,
        dynamics=search_dynamics,
        node_direct_evaluator=node_evaluator,
        index_computation=args.index_computation,
        depth_index=search_factory.depth_index,
    )

    return _TreeAndValueRuntimeDependencies(
        tree_manager=tree_manager,
        tree_factory=tree_factory,
        node_selector_create=search_factory.create_node_selector_factory(),
    )


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
) -> TreeExploration[AlgorithmNode[StateT]]:
    """Build and return a ready-to-run ``TreeExploration`` runtime.

    This is the preferred top-level construction path when callers want direct
    control over one search instance via ``step()`` or ``explore(...)``.
    ``create_search(...)`` is the shorter public alias for this constructor.
    """
    _ = state_type

    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT]
    node_tree_evaluation_factory = NodeTreeMinmaxEvaluationFactory[StateT]()

    return create_tree_and_value_exploration_with_tree_eval_factory(
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
) -> TreeExploration[AlgorithmNode[StateT]]:
    """Build a ready-to-run ``TreeExploration`` with an explicit value family.

    ``create_search_with_tree_eval_factory(...)`` is the shorter public alias
    for this family-aware constructor.
    """
    _ = state_type

    dependencies = _assemble_tree_and_value_runtime_dependencies(
        dynamics=dynamics,
        args=args,
        random_generator=random_generator,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=node_tree_evaluation_factory,
        hooks=hooks,
    )

    return create_tree_exploration(
        node_selector_create=dependencies.node_selector_create,
        starting_state=starting_state,
        tree_manager=dependencies.tree_manager,
        tree_factory=dependencies.tree_factory,
        stopping_criterion_args=args.stopping_criterion,
        recommend_branch_after_exploration=args.recommender_rule,
        notify_percent_function=notify_progress,
    )


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
    """Create a recommend-only wrapper around the tree-exploration runtime.

    Callers that want direct runtime control should prefer
    ``create_search(...)`` or ``create_tree_and_value_exploration(...)``. This
    wrapper remains useful when the only public operation needed is
    ``recommend(...)``.
    """
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
    """Create the recommend-only wrapper while reusing the main assembly path."""
    _ = state_type

    dependencies = _assemble_tree_and_value_runtime_dependencies(
        dynamics=dynamics,
        args=args,
        random_generator=random_generator,
        master_state_evaluator=master_state_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=node_tree_evaluation_factory,
        hooks=hooks,
    )

    return TreeAndValueBranchSelector(
        tree_manager=dependencies.tree_manager,
        random_generator=random_generator,
        tree_factory=dependencies.tree_factory,
        node_selector_create=dependencies.node_selector_create,
        stopping_criterion_args=args.stopping_criterion,
        recommend_branch_after_exploration=args.recommender_rule,
    )


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
) -> SearchRuntime[AlgorithmNode[StateT]]:
    """Preferred top-level constructor for a runnable search runtime."""
    return create_tree_and_value_exploration(
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


def create_search_with_tree_eval_factory[StateT: AnyTurnState, ActionT: Hashable](
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
) -> SearchRuntime[AlgorithmNode[StateT]]:
    """Preferred family-aware constructor for a runnable search runtime."""
    return create_tree_and_value_exploration_with_tree_eval_factory(
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
