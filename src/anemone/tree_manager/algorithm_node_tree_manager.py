"""Algorithm-aware tree-side phases used by ``TreeExploration``."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from anemone import trees
from anemone.dynamics import SearchDynamics
from anemone.indices.index_manager import (
    NodeExplorationIndexManager,
)
from anemone.indices.index_manager.node_exploration_manager import (
    update_all_indices,
)
from anemone.node_evaluation.common.branch_frontier import (
    require_branch_frontier_aware,
)
from anemone.node_evaluation.direct import (
    EvaluationQueries,
    NodeDirectEvaluator,
)
from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
)
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from anemone.updates.depth_index_propagator import DepthIndexPropagator
from anemone.updates.value_propagator import ValuePropagator

from .tree_expander import TreeExpansions
from .tree_manager import TreeManager


class BestLinePrintable(Protocol):
    """Evaluation capability for rendering the current best line."""

    def print_best_line(self) -> None:
        """Print the best line from the current node."""
        ...


@dataclass(slots=True)
class _ExpansionDirectEvaluation[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Direct-evaluation unit for nodes created by structural expansion."""

    evaluation_queries: EvaluationQueries
    get_node_evaluator: Callable[[], NodeDirectEvaluator | None]

    def evaluate_expansions(self, tree_expansions: TreeExpansions[NodeT]) -> None:
        """Stage newly created children and run direct evaluation."""
        node_evaluator = self.get_node_evaluator()
        assert node_evaluator is not None
        for node_to_evaluate in self._created_nodes_for_direct_evaluation(
            tree_expansions=tree_expansions
        ):
            node_evaluator.add_evaluation_query(
                node=node_to_evaluate,
                evaluation_queries=self.evaluation_queries,
            )

        node_evaluator.evaluate_all_queried_nodes(
            evaluation_queries=self.evaluation_queries
        )

    def _created_nodes_for_direct_evaluation(
        self,
        *,
        tree_expansions: TreeExpansions[NodeT],
    ) -> list[NodeT]:
        """Return the newly created child nodes that still need direct evaluation."""
        return [
            tree_expansion.child_node
            for tree_expansion in tree_expansions.expansions_with_node_creation
        ]


@dataclass(slots=True)
class _ExpansionPropagation[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Propagation unit deriving value/depth seeds from structural expansions."""

    get_value_propagator: Callable[[], ValuePropagator]
    get_depth_index_propagator: Callable[[], DepthIndexPropagator | None]

    def propagate_value_changes(self, tree_expansions: TreeExpansions[NodeT]) -> None:
        """Kick off upward value propagation from structural expansion records."""
        propagation_seed_nodes = self._propagation_seed_nodes(
            tree_expansions=tree_expansions
        )
        if not propagation_seed_nodes:
            return

        self.get_value_propagator().propagate_from_changed_nodes(propagation_seed_nodes)

    def propagate_depth_index(self, tree_expansions: TreeExpansions[NodeT]) -> None:
        """Kick off descendant-depth propagation from structural expansion records."""
        depth_index_propagator = self.get_depth_index_propagator()
        if depth_index_propagator is None:
            return

        propagation_seed_nodes = self._propagation_seed_nodes(
            tree_expansions=tree_expansions
        )
        if not propagation_seed_nodes:
            return

        depth_index_propagator.propagate_from_changed_nodes(propagation_seed_nodes)

    def _propagation_seed_nodes(
        self,
        *,
        tree_expansions: TreeExpansions[NodeT],
    ) -> list[NodeT]:
        """Return the child nodes that seed the post-expansion propagation waves."""
        return [tree_expansion.child_node for tree_expansion in tree_expansions]


@dataclass
class AlgorithmNodeTreeManager[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Provide algorithm-aware tree phases while ``TreeManager`` stays structural.

    ``TreeExploration`` owns the order of one search step. This collaborator
    provides the algorithm-aware phases that follow structural expansion:
    direct evaluation, upward propagation, descendant-depth propagation, and
    exploration-index refresh.
    """

    tree_manager: TreeManager[NodeT]

    evaluation_queries: EvaluationQueries
    node_evaluator: NodeDirectEvaluator | None
    index_manager: NodeExplorationIndexManager
    value_propagator: ValuePropagator = field(default_factory=ValuePropagator)
    depth_index_propagator: DepthIndexPropagator | None = None
    _direct_evaluation: _ExpansionDirectEvaluation[NodeT] = field(
        init=False,
        repr=False,
    )
    _propagation: _ExpansionPropagation[NodeT] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Assemble the narrow post-expansion units used by this phase API."""
        self._direct_evaluation = _ExpansionDirectEvaluation(
            evaluation_queries=self.evaluation_queries,
            get_node_evaluator=lambda: self.node_evaluator,
        )
        self._propagation = _ExpansionPropagation(
            get_value_propagator=lambda: self.value_propagator,
            get_depth_index_propagator=lambda: self.depth_index_propagator,
        )

    @property
    def dynamics(self) -> SearchDynamics[Any, Any]:
        """Return the search dynamics used by the wrapped tree manager."""
        return self.tree_manager.dynamics

    # Structural expansion phase
    def expand_instructions(
        self,
        tree: trees.Tree[NodeT],
        opening_instructions: OpeningInstructions[NodeT],
    ) -> TreeExpansions[NodeT]:
        """Perform algorithm-aware pre-expansion bookkeeping, then expand structurally."""
        self._mark_branches_opened(opening_instructions)
        return self.tree_manager.expand_instructions(
            tree=tree,
            opening_instructions=opening_instructions,
        )

    # Direct evaluation phase
    def evaluate_expansions(self, tree_expansions: TreeExpansions[NodeT]) -> None:
        """Delegate post-expansion direct evaluation to the direct-evaluation unit."""
        self._direct_evaluation.evaluate_expansions(tree_expansions)

    # Upward propagation phases
    def update_backward(self, tree_expansions: TreeExpansions[NodeT]) -> None:
        """Delegate post-expansion value propagation to the propagation unit."""
        self._propagation.propagate_value_changes(tree_expansions)

    def propagate_depth_index(self, tree_expansions: TreeExpansions[NodeT]) -> None:
        """Delegate post-expansion depth propagation to the propagation unit."""
        self._propagation.propagate_depth_index(tree_expansions)

    # Exploration-index refresh phase
    def refresh_exploration_indices(self, tree: trees.Tree[NodeT]) -> None:
        """Refresh exploration indices after the upward propagation phases."""
        update_all_indices(index_manager=self.index_manager, tree=tree)

    # Presentation helpers
    def print_some_stats(self, tree: trees.Tree[NodeT]) -> None:
        """Print statistics about the given tree.

        Args:
            tree: The tree to print statistics for.

        Returns:
            None

        """
        self.tree_manager.print_some_stats(tree=tree)

    def print_best_line(self, tree: trees.Tree[NodeT]) -> None:
        """Print the best branch line based on the current tree evaluation.

        Args:
            tree: The tree containing branches and their evaluations.

        """
        cast("BestLinePrintable", tree.root_node.tree_evaluation).print_best_line()

    # Private helpers
    def _mark_branches_opened(
        self,
        opening_instructions: OpeningInstructions[NodeT],
    ) -> None:
        """Update branch-frontier state before the structural expander mutates the tree."""
        opening_instruction: OpeningInstruction[NodeT]
        for opening_instruction in opening_instructions.values():
            require_branch_frontier_aware(
                opening_instruction.node_to_open.tree_evaluation
            ).on_branch_opened(opening_instruction.branch)
