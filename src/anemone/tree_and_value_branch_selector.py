"""Convenience wrapper for callers that only need ``recommend(...)``."""

from dataclasses import dataclass
from random import Random

from valanga.policy import NotifyProgressCallable, Recommendation

from anemone._valanga_types import AnyTurnState
from anemone.basics import Seed
from anemone.progress_monitor.progress_monitor import (
    AllStoppingCriterionArgs,
)
from anemone.search_factory import NodeSelectorFactory

from . import recommender_rule
from . import tree_manager as tree_man
from .tree_exploration import (
    TreeExploration,
    create_tree_exploration,
)
from .trees.factory import ValueTreeFactory


@dataclass
class TreeAndValueBranchSelector[StateT: AnyTurnState = AnyTurnState]:
    """Convenience API that builds and runs a fresh ``TreeExploration``.

    ``TreeExploration`` is the real runtime object. This wrapper remains useful
    when callers only want the one-shot ``recommend(...)`` interface instead of
    driving the runtime directly. ``SearchRecommender`` is the preferred public
    alias for this secondary concept.
    """

    # pretty empty class but might be useful when dealing with multi round and time , no?

    tree_manager: tree_man.AlgorithmNodeTreeManager
    tree_factory: ValueTreeFactory[StateT]
    stopping_criterion_args: AllStoppingCriterionArgs
    node_selector_create: NodeSelectorFactory
    random_generator: Random
    recommend_branch_after_exploration: recommender_rule.AllRecommendFunctionsArgs

    def recommend(
        self,
        state: StateT,
        seed: Seed,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> Recommendation:
        """Build a runtime, explore from ``state``, and return the recommendation.

        Args:
            state (StateT): The current state to explore.
            seed (Seed): The seed used for randomization during branch selection.
            notify_progress (NotifyProgressCallable | None): Optional progress callback.

        Returns:
            Recommendation: The recommended branch based on the tree and value strategy.

        """
        tree_exploration: TreeExploration = self.create_tree_exploration(
            state=state, notify_progress=notify_progress
        )
        self.random_generator.seed(seed)

        branch_recommendation: Recommendation = tree_exploration.explore(
            random_generator=self.random_generator
        ).branch_recommendation

        return branch_recommendation

    def create_tree_exploration(
        self,
        state: StateT,
        notify_progress: NotifyProgressCallable | None = None,
    ) -> TreeExploration:
        """Build a fresh runnable ``TreeExploration`` for the given state."""
        tree_exploration: TreeExploration = create_tree_exploration(
            tree_manager=self.tree_manager,
            node_selector_create=self.node_selector_create,
            starting_state=state,
            tree_factory=self.tree_factory,
            stopping_criterion_args=self.stopping_criterion_args,
            recommend_branch_after_exploration=self.recommend_branch_after_exploration,
            notify_percent_function=notify_progress,
        )
        return tree_exploration

    def print_info(self) -> None:
        """Print information about the branch selector type."""
        print("type: Tree and Value")


SearchRecommender = TreeAndValueBranchSelector
