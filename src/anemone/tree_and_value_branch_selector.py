"""
This module contains the implementation of the TreeAndValueMoveSelector class, which is responsible for selecting branches
based on a tree and value strategy.

The TreeAndValueMoveSelector class uses a tree-based approach to explore possible branches and select the best branch based on
a value function. It utilizes a tree manager, a tree factory, stopping criterion arguments, a node selector factory, a
random generator, and recommendation functions to guide the branch selection process.

The TreeAndValueMoveSelector class provides the following methods:
- select_branch: Selects the best branch based on the tree and value strategy.
- print_info: Prints information about the branch selector type.
"""

from dataclasses import dataclass
from queue import Queue
from random import Random

from valanga import TurnState

from anemone.basics import BranchRecommendation, Seed
from anemone.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
)
from anemone.search_factory import NodeSelectorFactory
from anemone.utils.dataclass import IsDataclass

from . import recommender_rule
from . import tree_manager as tree_man
from .tree_exploration import TreeExploration, create_tree_exploration
from .trees.factory import ValueTreeFactory


@dataclass
class TreeAndValueBranchSelector[StateT: TurnState = TurnState]:
    """
    The TreeAndValueBranchSelector class is responsible for selecting branches based on a tree and value strategy.

    Attributes:
    - tree_manager: The tree manager responsible for managing the algorithm nodes.
    - tree_factory: The tree factory responsible for creating branch and value trees.
    - stopping_criterion_args: The stopping criterion arguments used to determine when to stop the tree exploration.
    - node_selector_create: The node selector factory used to create node selectors for tree exploration.
    - random_generator: The random generator used for randomization during tree exploration.
    - recommend_branch_after_exploration: The recommendation functions used to recommend a branch after tree exploration.
    """

    # pretty empty class but might be useful when dealing with multi round and time , no?

    tree_manager: tree_man.AlgorithmNodeTreeManager
    tree_factory: ValueTreeFactory[StateT]
    stopping_criterion_args: AllStoppingCriterionArgs
    node_selector_create: NodeSelectorFactory
    random_generator: Random
    recommend_branch_after_exploration: recommender_rule.AllRecommendFunctionsArgs
    queue_progress_player: Queue[IsDataclass] | None

    def select_branch(
        self, state: StateT, selection_seed: Seed
    ) -> BranchRecommendation:
        """
        Selects the best branch based on the tree and value strategy.

        Args:
        - board: The current board state.
        - selection_seed: The seed used for randomization during branch selection.

        Returns:
        - The recommended branch based on the tree and value strategy.
        """
        tree_exploration: TreeExploration = self.create_tree_exploration(state=state)
        self.random_generator.seed(selection_seed)

        branch_recommendation: BranchRecommendation = tree_exploration.explore(
            random_generator=self.random_generator
        ).branch_recommendation

        return branch_recommendation

    def create_tree_exploration(
        self,
        state: StateT,
    ) -> TreeExploration:
        """Create a TreeExploration instance for the given state."""
        tree_exploration: TreeExploration = create_tree_exploration(
            tree_manager=self.tree_manager,
            node_selector_create=self.node_selector_create,
            starting_state=state,
            tree_factory=self.tree_factory,
            stopping_criterion_args=self.stopping_criterion_args,
            recommend_branch_after_exploration=self.recommend_branch_after_exploration,
            queue_progress_player=self.queue_progress_player,
        )
        return tree_exploration

    def print_info(self) -> None:
        """
        Prints information about the branch selector type.
        """
        print("type: Tree and Value")
