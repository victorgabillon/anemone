"""Runnable tree-search runtime centered on ``TreeExploration``."""

from collections.abc import Callable
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING, Any

from valanga import BranchKey, State
from valanga.evaluations import Value
from valanga.game import BranchName
from valanga.policy import NotifyProgressCallable, Recommendation

from anemone._valanga_types import AnyTurnState
from anemone.dynamics import SearchDynamics
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
    ProgressMonitor,
    create_stopping_criterion,
)
from anemone.search_factory import NodeSelectorFactory

from . import node_selector as node_sel
from . import recommender_rule, trees
from . import tree_manager as tree_man
from .trees.factory import ValueTreeFactory

if TYPE_CHECKING:
    from valanga.policy import BranchPolicy


type IterationProgressReporter = Callable[[Any, Random], None]
type SearchResultReporter = Callable[[Any], None]


@dataclass
class TreeExplorationResult[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Result of exploring one search tree."""

    branch_recommendation: Recommendation
    tree: trees.Tree[NodeT]


def compute_child_evals[StateT: State](
    root: AlgorithmNode[StateT],
    dynamics: SearchDynamics[StateT, Any],
) -> dict[BranchName, Value]:
    """Compute evaluations for each existing child branch."""
    evals: dict[BranchName, Value] = {}
    for bk, child in root.branches_children.items():
        if child is None:
            continue

        # Use whatever your canonical per-node evaluation is:
        bk_name = dynamics.action_name(root.state, bk)
        evals[bk_name] = child.tree_evaluation.get_value()
    return evals


@dataclass
class TreeExploration[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Runnable runtime object for one tree search.

    ``TreeExploration`` is the single explicit sequencing owner for one search
    iteration. It owns the search state for one run and exposes the two main
    runtime operations:

    * ``step()`` for one iteration
    * ``explore(...)`` for a full run to recommendation

    ``SearchRuntime`` is the preferred public alias for callers who want a
    shorter top-level name for this runtime concept.
    """

    tree: trees.Tree[NodeT]
    tree_manager: tree_man.AlgorithmNodeTreeManager[NodeT]
    node_selector: node_sel.NodeSelector[NodeT]
    recommend_branch_after_exploration: recommender_rule.AllRecommendFunctionsArgs
    stopping_criterion: ProgressMonitor[NodeT]
    notify_percent_function: NotifyProgressCallable
    iteration_progress_reporter: IterationProgressReporter | None = None
    search_result_reporter: SearchResultReporter | None = None
    _latest_tree_expansions: tree_man.TreeExpansions[NodeT] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Seed selector-visible iteration state for the first step."""
        self._latest_tree_expansions = self._make_initial_tree_expansions()

    def _make_initial_tree_expansions(self) -> tree_man.TreeExpansions[NodeT]:
        """Return the synthetic root creation log used before the first step."""
        tree_expansions: tree_man.TreeExpansions[NodeT] = tree_man.TreeExpansions()
        tree_expansions.record_creation(
            tree_expansion=tree_man.TreeExpansion(
                child_node=self.tree.root_node,
                parent_node=None,
                state_modifications=None,
                creation_child_node=True,
                branch_key=None,
            )
        )
        return tree_expansions

    def _report_iteration_progress(self, random_generator: Random) -> None:
        """Invoke the optional iteration-progress reporter for this runtime."""
        if self.iteration_progress_reporter is None:
            return
        self.iteration_progress_reporter(self, random_generator)

    def print_info_during_branch_computation(self, random_generator: Random) -> None:
        """Backward-compatible alias for the iteration-progress reporter."""
        self._report_iteration_progress(random_generator)

    def _report_search_result(self) -> None:
        """Invoke the optional search-result reporter for this runtime."""
        if self.search_result_reporter is None:
            return
        self.search_result_reporter(self)

    def _select_node_for_expansion(self) -> node_sel.OpeningInstructions[NodeT]:
        """Ask the selector for the next branches to open."""
        return self.node_selector.choose_node_and_branch_to_open(
            tree=self.tree,
            latest_tree_expansions=self._latest_tree_expansions,
        )

    def _limit_opening_instructions(
        self,
        opening_instructions: node_sel.OpeningInstructions[NodeT],
    ) -> node_sel.OpeningInstructions[NodeT]:
        """Apply the stopping criterion's opening-budget filter."""
        return self.stopping_criterion.respectful_opening_instructions(
            opening_instructions=opening_instructions,
            tree=self.tree,
        )

    def _expand_opening_instructions(
        self,
        opening_instructions: node_sel.OpeningInstructions[NodeT],
    ) -> tree_man.TreeExpansions[NodeT]:
        """Run the structural expansion phase for one iteration."""
        return self.tree_manager.expand_instructions(
            tree=self.tree,
            opening_instructions=opening_instructions,
        )

    def _evaluate_expansions(
        self,
        tree_expansions: tree_man.TreeExpansions[NodeT],
    ) -> None:
        """Run the direct-evaluation phase for newly created nodes."""
        self.tree_manager.evaluate_expansions(tree_expansions=tree_expansions)

    def _propagate_iteration_updates(
        self,
        tree_expansions: tree_man.TreeExpansions[NodeT],
    ) -> None:
        """Run the post-evaluation propagation and refresh phases."""
        self.tree_manager.update_backward(tree_expansions=tree_expansions)
        self.tree_manager.propagate_depth_index(tree_expansions=tree_expansions)
        self.tree_manager.refresh_exploration_indices(tree=self.tree)

    def step(self) -> None:
        """Run one search iteration in the canonical runtime order."""
        # Search stops once the root value is exact, even if the root state is
        # still non-terminal and some siblings remain unopened.
        assert not self.tree.root_node.tree_evaluation.has_exact_value()

        opening_instructions = self._select_node_for_expansion()
        opening_instructions_subset = self._limit_opening_instructions(
            opening_instructions
        )
        tree_expansions = self._expand_opening_instructions(opening_instructions_subset)
        self._evaluate_expansions(tree_expansions)
        self._propagate_iteration_updates(tree_expansions)
        self._latest_tree_expansions = tree_expansions

    def explore(self, random_generator: Random) -> TreeExplorationResult[NodeT]:
        """Explore the tree to find the best branch.

        Args:
            random_generator (Random): The random number generator.

        Returns:
            TreeExplorationResult[NodeT]: The recommended branch and its evaluation.

        """
        self._latest_tree_expansions = self._make_initial_tree_expansions()

        loop: int = 0
        while self.stopping_criterion.should_we_continue(tree=self.tree):
            loop = loop + 1
            self._report_iteration_progress(random_generator=random_generator)
            self.step()

            if loop % 10 == 0:
                self.stopping_criterion.notify_percent_progress(
                    tree=self.tree, notify_percent_function=self.notify_percent_function
                )

        policy: BranchPolicy = self.recommend_branch_after_exploration.policy(
            self.tree.root_node
        )

        best_branch: BranchKey = self.recommend_branch_after_exploration.sample(
            policy, random_generator
        )

        best_branch_name = self.tree_manager.dynamics.action_name(
            self.tree.root_node.state, best_branch
        )
        self._report_search_result()

        branch_recommendation = Recommendation(
            recommended_name=best_branch_name,
            evaluation=self.tree.root_node.tree_evaluation.get_value(),
            policy=policy,
            branch_evals=compute_child_evals(
                self.tree.root_node,
                dynamics=self.tree_manager.dynamics,
            ),
        )

        tree_exploration_result: TreeExplorationResult[NodeT] = TreeExplorationResult(
            branch_recommendation=branch_recommendation, tree=self.tree
        )

        return tree_exploration_result


SearchRuntime = TreeExploration


def create_tree_exploration[StateT: AnyTurnState](
    node_selector_create: NodeSelectorFactory,
    starting_state: StateT,
    tree_manager: tree_man.AlgorithmNodeTreeManager[AlgorithmNode[StateT]],
    tree_factory: ValueTreeFactory[StateT],
    stopping_criterion_args: AllStoppingCriterionArgs,
    recommend_branch_after_exploration: recommender_rule.AllRecommendFunctionsArgs,
    notify_percent_function: NotifyProgressCallable | None = None,
    iteration_progress_reporter: IterationProgressReporter | None = None,
    search_result_reporter: SearchResultReporter | None = None,
) -> TreeExploration[AlgorithmNode[StateT]]:
    """Assemble ``TreeExploration`` from already-created collaborators.

    Top-level callers usually prefer the higher-level
    ``anemone.create_tree_and_value_exploration(...)`` helpers, which build
    these collaborators and return this same runtime object directly.
    """
    # creates the tree
    tree: trees.Tree[AlgorithmNode[StateT]] = tree_factory.create(
        starting_state=starting_state
    )
    # creates the node selector
    node_selector: node_sel.NodeSelector[AlgorithmNode[StateT]] = node_selector_create()
    stopping_criterion: ProgressMonitor[AlgorithmNode[StateT]] = (
        create_stopping_criterion(
            args=stopping_criterion_args, node_selector=node_selector
        )
    )

    tree_exploration: TreeExploration[AlgorithmNode[StateT]] = TreeExploration(
        tree=tree,
        tree_manager=tree_manager,
        stopping_criterion=stopping_criterion,
        node_selector=node_selector,
        recommend_branch_after_exploration=recommend_branch_after_exploration,
        notify_percent_function=notify_percent_function,
        iteration_progress_reporter=iteration_progress_reporter,
        search_result_reporter=search_result_reporter,
    )

    return tree_exploration
