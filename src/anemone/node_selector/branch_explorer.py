"""This module contains the branchExplorer class and its subclasses.
branchExplorer is responsible for exploring branches in a game tree.
"""

from enum import StrEnum
from random import Random
from typing import Any

from valanga import BranchKey

from anemone.node_selector.notations_and_statics import (
    zipf_picks_random,
)
from anemone.nodes.algorithm_node import AlgorithmNode


class SamplingPriorities(StrEnum):
    """Enumeration class representing the sampling priorities for branch exploration.

    Attributes:
        NO_PRIORITY (str): No priority for branch sampling.
        PRIORITY_BEST (str): Priority for the best branch.
        PRIORITY_TWO_BEST (str): Priority for the two best branches.

    """

    NO_PRIORITY = "no_priority"
    PRIORITY_BEST = "priority_best"
    PRIORITY_TWO_BEST = "priority_two_best"


class BranchExplorer:
    """BranchExplorer is responsible for exploring branches in a game tree.
    It provides a method to sample a child node to explore.
    """

    priority_sampling: SamplingPriorities

    def __init__(self, priority_sampling: SamplingPriorities) -> None:
        """Initializes a branchExplorer instance.

        Args:
            priority_sampling (SamplingPriorities): The priority sampling strategy to use.

        """
        self.priority_sampling = priority_sampling


class ZipfBranchExplorer(BranchExplorer):
    """ZipfBranchExplorer is a subclass of BranchExplorer that uses the Zipf distribution for sampling."""

    def __init__(
        self, priority_sampling: SamplingPriorities, random_generator: Random
    ) -> None:
        """Initializes a ZipfbranchExplorer instance.

        Args:
            priority_sampling (SamplingPriorities): The priority sampling strategy to use.
            random_generator (Random): The random number generator to use.

        """
        super().__init__(priority_sampling)
        self.random_generator = random_generator

    def sample_branch_to_explore(
        self, tree_node_to_sample_from: AlgorithmNode[Any]
    ) -> BranchKey:
        """Samples a child node to explore from the given tree node.

        Args:
            tree_node_to_sample_from (AlgorithmNode): The tree node to sample from.

        Returns:
            AlgorithmNode: The sampled child node to explore.

        """
        sorted_not_over_branches: list[BranchKey] = (
            tree_node_to_sample_from.tree_evaluation.sort_branches_not_over()
        )

        return zipf_picks_random(
            ordered_list_elements=sorted_not_over_branches,
            random_generator=self.random_generator,
        )
