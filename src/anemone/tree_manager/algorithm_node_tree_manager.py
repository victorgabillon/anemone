"""Algorithm-aware tree-side phases used by ``TreeExploration``."""

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
        """Evaluate newly created child nodes from structural expansion records."""
        assert self.node_evaluator is not None
        for node_to_evaluate in self._created_nodes_for_direct_evaluation(
            tree_expansions=tree_expansions
        ):
            self.node_evaluator.add_evaluation_query(
                node=node_to_evaluate,
                evaluation_queries=self.evaluation_queries,
            )

        self.node_evaluator.evaluate_all_queried_nodes(
            evaluation_queries=self.evaluation_queries
        )

    # Upward propagation phases
    def update_backward(self, tree_expansions: TreeExpansions[NodeT]) -> None:
        """Kick off upward value propagation from the latest structural changes."""
        propagation_seed_nodes = self._propagation_seed_nodes(
            tree_expansions=tree_expansions
        )
        if not propagation_seed_nodes:
            return

        self.value_propagator.propagate_from_changed_nodes(propagation_seed_nodes)

    def propagate_depth_index(self, tree_expansions: TreeExpansions[NodeT]) -> None:
        """Kick off descendant-depth propagation from the latest structural changes."""
        if self.depth_index_propagator is None:
            return

        propagation_seed_nodes = self._propagation_seed_nodes(
            tree_expansions=tree_expansions
        )
        if not propagation_seed_nodes:
            return

        self.depth_index_propagator.propagate_from_changed_nodes(propagation_seed_nodes)

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

    def _propagation_seed_nodes(
        self,
        *,
        tree_expansions: TreeExpansions[NodeT],
    ) -> list[NodeT]:
        """Return the child nodes that seed the upward propagation phases.

        Each structural expansion record names the child node whose changed
        structural/value state parents must observe next. That makes the child
        node the correct seed for both value propagation and descendant-depth
        propagation.
        """
        return [tree_expansion.child_node for tree_expansion in tree_expansions]

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
