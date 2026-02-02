"""
This module contains the implementation of the TreeExploration class, which is responsible for managing a search
 for the best branch in a given state using a tree-based approach.

The TreeExploration class is used to create and manage a tree structure that represents the possible branches and
 their evaluations in a state space. It provides methods for exploring the tree, selecting the best branch,
  and printing information during the branch computation.

The module also includes helper functions for creating a TreeExploration object and its dependencies.

Classes:
- TreeExploration: Manages the search for the best branch using a tree-based approach.

Functions:
- create_tree_exploration: Creates a TreeExploration object with the specified dependencies.
"""

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any

from valanga import BranchKey, State, StateEvaluation, TurnState
from valanga.game import BranchName
from valanga.policy import Recommendation

from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
    ProgressMonitor,
    create_stopping_criterion,
)
from anemone.search_factory import NodeSelectorFactory
from anemone.utils.logger import anemone_logger

from . import node_selector as node_sel
from . import recommender_rule, trees
from . import tree_manager as tree_man
from .trees.factory import ValueTreeFactory

if TYPE_CHECKING:
    from valanga.policy import BranchPolicy

from valanga.policy import NotifyProgressCallable

@dataclass
class TreeExplorationResult[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """
    Tree Exploration Result holds the result of a tree exploration.
    """

    branch_recommendation: Recommendation
    tree: trees.Tree[NodeT]


def compute_child_evals[StateT: State](
    root: AlgorithmNode[StateT],
) -> dict[BranchName, StateEvaluation]:
    """Compute evaluations for each existing child branch."""
    evals: dict[BranchName, StateEvaluation] = {}
    for bk, child in root.branches_children.items():
        if child is None:
            continue

        # Use whatever your canonical per-node evaluation is:
        bk_name = root.state.branch_name_from_key(bk)
        evals[bk_name] = child.tree_evaluation.evaluate()
    return evals


@dataclass
class TreeExploration[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """
    Tree Exploration is an object to manage one best-branch search.

    Attributes:
    - tree: The tree structure representing the possible branches and their evaluations.
    - tree_manager: The manager for the tree structure.
    - node_selector: The selector for choosing nodes and branches to open in the tree.
    - recommend_branch_after_exploration: The recommender rule for selecting the best branch after the exploration.
    - stopping_criterion: The stopping criterion for determining when to stop the tree exploration.

    Methods:
    - print_info_during_branch_computation: Prints information during the branch computation.
    - explore: Explores the tree to find the best branch.
    """

    # TODO Not sure why this class is not simply the TreeAndValuePlayer Class
    #  but might be useful when dealing with multi round and time , no?

    tree: trees.Tree[NodeT]
    tree_manager: tree_man.AlgorithmNodeTreeManager[NodeT]
    node_selector: node_sel.NodeSelector[NodeT]
    recommend_branch_after_exploration: recommender_rule.AllRecommendFunctionsArgs
    stopping_criterion: ProgressMonitor[NodeT]
    notify_percent_function: NotifyProgressCallable

    def print_info_during_branch_computation(self, random_generator: Random) -> None:
        """Print info during the branch computation.
        Args:
        - random_generator: The random number generator.
        """
        current_best_branch: str
        if self.tree.root_node.tree_evaluation.best_branch_sequence:
            current_best_branch = str(
                self.tree.root_node.tree_evaluation.best_branch_sequence[0]
            )
        else:
            current_best_branch = "?"
        if random_generator.random() < 0.11:
            anemone_logger.info(
                "state: %s",
                self.tree.root_node.state,
            )

            str_progress = self.stopping_criterion.get_string_of_progress(self.tree)
            anemone_logger.info(
                "%s | current best branch: %s | current white value: %s",
                str_progress,
                current_best_branch,
                self.tree.root_node.tree_evaluation.value_white_minmax,
            )

            # ,end='\r')
            self.tree.root_node.tree_evaluation.print_branches_sorted_by_value_and_exploration()
            self.tree_manager.print_best_line(tree=self.tree)

    def explore(self, random_generator: Random) -> TreeExplorationResult[NodeT]:
        """
        Explore the tree to find the best branch.

        Args:
        - random_generator: The random number generator.

        Returns:
        - BranchRecommendation: The recommended branch and its evaluation.
        """
        # by default the first tree expansion is the creation of the tree node
        tree_expansions: tree_man.TreeExpansions[NodeT] = tree_man.TreeExpansions()

        tree_expansion: tree_man.TreeExpansion[NodeT] = tree_man.TreeExpansion(
            child_node=self.tree.root_node,
            parent_node=None,
            state_modifications=None,
            creation_child_node=True,
            branch_key=None,
        )
        tree_expansions.add_creation(tree_expansion=tree_expansion)

        loop: int = 0
        while self.stopping_criterion.should_we_continue(tree=self.tree):
            loop = loop + 1
            assert not self.tree.root_node.is_over()
            # print info
            self.print_info_during_branch_computation(random_generator=random_generator)

            # choose the branches and nodes to open
            opening_instructions: node_sel.OpeningInstructions[NodeT]
            opening_instructions = self.node_selector.choose_node_and_branch_to_open(
                tree=self.tree, latest_tree_expansions=tree_expansions
            )

            # make sure we do not break the stopping criterion
            opening_instructions_subset: node_sel.OpeningInstructions[NodeT]
            opening_instructions_subset = (
                self.stopping_criterion.respectful_opening_instructions(
                    opening_instructions=opening_instructions, tree=self.tree
                )
            )

            # open the nodes
            tree_expansions = self.tree_manager.open_instructions(
                tree=self.tree, opening_instructions=opening_instructions_subset
            )

            # self.node_selector.communicate_expansions()
            self.tree_manager.update_backward(tree_expansions=tree_expansions)
            self.tree_manager.update_indices(tree=self.tree)

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

        best_branch_name = self.tree.root_node.state.branch_name_from_key(best_branch)
        self.tree_manager.print_best_line(
            tree=self.tree
        )  # todo maybe almost best chosen line no?

        branch_recommendation = Recommendation(
            recommended_name=best_branch_name,
            evaluation=self.tree.root_node.tree_evaluation.evaluate(),
            policy=policy,
            branch_evals=compute_child_evals(self.tree.root_node),
        )

        tree_exploration_result: TreeExplorationResult[NodeT] = TreeExplorationResult(
            branch_recommendation=branch_recommendation, tree=self.tree
        )

        return tree_exploration_result


def create_tree_exploration[StateT: TurnState](
    node_selector_create: NodeSelectorFactory,
    starting_state: StateT,
    tree_manager: tree_man.AlgorithmNodeTreeManager[AlgorithmNode[StateT]],
    tree_factory: ValueTreeFactory[StateT],
    stopping_criterion_args: AllStoppingCriterionArgs,
    recommend_branch_after_exploration: recommender_rule.AllRecommendFunctionsArgs,
    notify_percent_function: NotifyProgressCallable | None = None,
) -> TreeExploration[AlgorithmNode[StateT]]:
    """
    Create a TreeExploration object with the specified dependencies.

    Args:
    - node_selector_create: The factory function for creating the node selector.
    - starting_state: The starting state for the exploration.
    - tree_manager: The manager for the tree structure.
    - tree_factory: The factory for creating the tree structure.
    - stopping_criterion_args: The arguments for creating the stopping criterion.
    - recommend_branch_after_exploration: The recommender rule for selecting the best branch after exploration.

    Returns:
    - TreeExploration: The created TreeExploration object.
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
    )

    return tree_exploration
