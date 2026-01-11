"""
This module contains the implementation of the RecurZipfBase class, which is a node selector for a move selector tree.

The RecurZipfBase class is responsible for selecting the next node to explore in a move selector tree based on the RecurZipf algorithm.

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
    """
    Arguments for the RecurZipfBase node selector.

    Attributes:
        move_explorer_priority (SamplingPriorities): The priority for move exploration.
    """

    type: Literal[NodeSelectorType.RECUR_ZIPF_BASE]
    branch_explorer_priority: SamplingPriorities


class RecurZipfBase[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """The RecurZipfBase Node selector"""

    opening_instructor: OpeningInstructor

    def __init__(
        self,
        args: RecurZipfBaseArgs,
        random_generator: Random,
        opening_instructor: OpeningInstructor,
    ) -> None:
        """
        Initializes a new instance of the RecurZipfBase class.

        Args:
        - args (RecurZipfBaseArgs): The arguments for the RecurZipfBase node selector.
        - random_generator (random.Random): The random number generator.
        - opening_instructor (OpeningInstructor): The opening instructor.

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
        """
        Chooses the next node to explore and the move to open.

        Args:
        - tree (trees.Tree[AlgorithmNode]): The move selector tree.
        - latest_tree_expansions (tree_man.TreeExpansions): The latest tree expansions.

        Returns:
        - OpeningInstructions: The instructions for opening the selected move.

        """
        # todo maybe proportions and proportions can be valuesorted dict with smart updates

        _ = latest_tree_expansions  # not used here
        opening_instructions: OpeningInstructions[NodeT]
        # TODO make sure this block is put in chipiron now with a wrapper
        # best_node_sequence = best_node_sequence_from_node(tree.root_node)
        # if best_node_sequence:
        #     last_node_in_best_line = best_node_sequence[-1]
        #     assert isinstance(last_node_in_best_line, AlgorithmNode)
        #     if (
        #         last_node_in_best_line.state.is_attacked(
        #             not last_node_in_best_line.tree_node.player_to_move
        #         )
        #         and not last_node_in_best_line.minmax_evaluation.is_over()
        #     ):
        #         # print('best line is underattacked')
        #         if self.random_generator.random() > 0.5:
        #             # print('best line is underattacked and i do')
        #             all_moves_to_open: list[BranchKey] = (
        #                 self.opening_instructor.all_branches_to_open(
        #                     node_to_open=last_node_in_best_line.tree_node
        #                 )
        #             )
        #             opening_instructions = create_instructions_to_open_all_branches(
        #                 branches_to_play=all_moves_to_open,
        #                 node_to_open=last_node_in_best_line,
        #             )
        #             return opening_instructions

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
        """
        Returns a string representation of the RecurZipfBase node selector.

        Returns:
        - str: The string representation of the RecurZipfBase node selector.

        """
        return "RecurZipfBase"
