"""Internal collaborator wiring for the public search-construction API.

This module assembles the concrete collaborators used by the public
constructors in ``anemone.factory``. It is intentionally implementation-
oriented and is not meant to be the primary user-facing API surface.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Protocol

from valanga import Dynamics, RepresentationFactory, StateModifications
from valanga.evaluator_types import EvaluatorInput

from anemone import node_factory
from anemone import search_factory as search_factories
from anemone._valanga_types import AnyTurnState
from anemone.dynamics import SearchDynamics, normalize_search_dynamics
from anemone.hooks.search_hooks import SearchHooks
from anemone.indices.node_indices.index_types import IndexComputationType
from anemone.node_evaluation.direct.factory import create_node_evaluator
from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
)
from anemone.node_evaluation.tree.factory import NodeTreeEvaluationFactory
from anemone.node_factory.base import TreeNodeFactory
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.opening_instructions import OpeningType
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.tree_manager import (
    AlgorithmNodeTreeManager,
    create_algorithm_node_tree_manager,
)
from anemone.trees.factory import ValueTreeFactory

if TYPE_CHECKING:
    from anemone.node_evaluation.direct.node_direct_evaluator import (
        NodeDirectEvaluator,
    )


class SearchBuildArgsP(Protocol):
    """Minimal config surface needed to assemble search collaborators."""

    node_selector: ComposedNodeSelectorArgs
    opening_type: OpeningType
    index_computation: IndexComputationType | None


@dataclass(frozen=True, slots=True)
class SearchRuntimeDependencies[StateT: AnyTurnState]:
    """Reusable collaborators needed to assemble one search runtime."""

    tree_manager: AlgorithmNodeTreeManager[AlgorithmNode[StateT]]
    tree_factory: ValueTreeFactory[StateT]
    node_selector_create: search_factories.NodeSelectorFactory


def assemble_search_runtime_dependencies[
    StateT: AnyTurnState,
    ActionT: Hashable,
](
    *,
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    args: SearchBuildArgsP,
    random_generator: Random,
    master_state_evaluator: MasterStateValueEvaluator,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None,
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT],
    hooks: SearchHooks | None = None,
) -> SearchRuntimeDependencies[StateT]:
    """Assemble the concrete collaborators behind one search runtime."""
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

    tree_manager = create_algorithm_node_tree_manager(
        algorithm_node_factory=algorithm_node_factory,
        dynamics=search_dynamics,
        node_direct_evaluator=node_evaluator,
        index_computation=args.index_computation,
        depth_index=search_factory.depth_index,
    )

    return SearchRuntimeDependencies(
        tree_manager=tree_manager,
        tree_factory=tree_factory,
        node_selector_create=search_factory.create_node_selector_factory(),
    )
