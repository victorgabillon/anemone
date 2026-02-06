"""Provide the implementation of the RecurZipfBase class for branch selector trees.

The RecurZipfBase class is responsible for selecting the next node to explore in a branch selector tree based on the RecurZipf algorithm.

Classes:
- RecurZipfBase: The RecurZipfBase Node selector.

"""

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any, Literal

from anemone import trees
from anemone.node_selector.branch_explorer import SamplingPriorities, ZipfBranchExplorer
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
    create_instructions_to_open_all_branches,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

if TYPE_CHECKING:
    from anemone import tree_manager as tree_man


@dataclass
class RecurZipfBaseArgs:
    """Arguments for the RecurZipfBase node selector.

    Attributes:
        branch_explorer_priority (SamplingPriorities): The priority for branch exploration.

    """

    type: Literal[NodeSelectorType.RECUR_ZIPF_BASE]
    branch_explorer_priority: SamplingPriorities


class RecurZipfBase[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """The RecurZipfBase Node selector."""

    opening_instructor: OpeningInstructor

    def __init__(
        self,
        args: RecurZipfBaseArgs,
        random_generator: Random,
        opening_instructor: OpeningInstructor,
    ) -> None:
        """Initialize a new instance of the RecurZipfBase class.

        Args:
            args (RecurZipfBaseArgs): Arguments for the node selector.
            random_generator (Random): Random number generator.
            opening_instructor (OpeningInstructor): Opening instructor.

        """
        self.opening_instructor = opening_instructor
        self.branch_explorer = ZipfBranchExplorer(
            args.branch_explorer_priority, random_generator
        )
        self.random_generator = random_generator

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> OpeningInstructions[NodeT]:
        """Choose the next node to explore and the branch to open.

        Args:
            tree (trees.Tree[AlgorithmNode]): The branch selector tree.
            latest_tree_expansions (tree_man.TreeExpansions): The latest tree expansions.

        Returns:
            OpeningInstructions: Instructions for opening the selected branch.

        """
        # TODO: maybe proportions and proportions can be valuesorted dict with smart updates

        _ = latest_tree_expansions  # not used here
        opening_instructions: OpeningInstructions[NodeT]

        wandering_node: NodeT = tree.root_node

        while wandering_node.tree_evaluation.branches_not_over:
            assert not wandering_node.is_over()
            branch = self.branch_explorer.sample_branch_to_explore(
                tree_node_to_sample_from=wandering_node
            )
            next_node = wandering_node.branches_children[branch]
            assert next_node is not None
            wandering_node = next_node

        all_branches_to_open = self.opening_instructor.all_branches_to_open(
            node_to_open=wandering_node
        )
        opening_instructions = create_instructions_to_open_all_branches(
            branches_to_play=all_branches_to_open, node_to_open=wandering_node
        )

        return opening_instructions

    def __str__(self) -> str:
        """Return a string representation of the RecurZipfBase node selector.

        Returns:
        - str: The string representation of the RecurZipfBase node selector.

        """
        return "RecurZipfBase"
