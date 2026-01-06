"""
This module contains the implementation of the Uniform Node selector.

The Uniform Node selector is responsible for selecting nodes to expand in a tree-based move selector algorithm.
It uses an opening instructor to determine the moves to open for each node and generates opening instructions accordingly.

Classes:
- Uniform: The Uniform Node selector class.

"""


from typing import Any

from valanga import BranchKey

from anemone import tree_manager as tree_man
from anemone import trees
from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
    create_instructions_to_open_all_branches,
)
from anemone.nodes.algorithm_node import AlgorithmNode
class Uniform[TNode: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """The Uniform Node selector"""

    opening_instructor: OpeningInstructor

    def __init__(self, opening_instructor: OpeningInstructor) -> None:
        """
        Initializes a new instance of the Uniform class.

        Args:
        - opening_instructor (OpeningInstructor): The opening instructor to be used for determining moves to open.

        """
        self.opening_instructor = opening_instructor
        self.current_depth_to_expand = 0

    def get_current_depth_to_expand(self) -> int:
        """
        Gets the current depth to expand.

        Returns:
        - int: The current depth to expand.

        """
        return self.current_depth_to_expand

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[TNode],
        latest_tree_expansions: tree_man.TreeExpansions[TNode],
    ) -> OpeningInstructions[TNode]:
        """
        Chooses a node to expand and determines the moves to open for that node.

        Args:
        - tree (trees.Tree[AlgorithmNode]): The move and value tree.
        - latest_tree_expansions (tree_man.TreeExpansions): The latest tree expansions.

        Returns:
        - OpeningInstructions: The opening instructions for the chosen node.

        """
        opening_instructions_batch: OpeningInstructions[TNode] = OpeningInstructions()

        # generate the nodes to expand
        current_half_move_to_expand = (
            tree.tree_root_tree_depth + self.current_depth_to_expand
        )

        # self.tree.descendants.print_info()
        nodes_to_consider = list(tree.descendants[current_half_move_to_expand].values())

        # filter the game-over ones and the ones with values
        nodes_to_consider_not_over: list[TNode] = [
            node
            for node in nodes_to_consider
            if not node.is_over()
        ]

        # sort them by order of importance for the player
        nodes_to_consider_sorted_by_value = sorted(
            nodes_to_consider_not_over,
            key=lambda x: tree.root_node.tree_evaluation.subjective_value_of(
                x.tree_evaluation
            ),
        )  # best last

        for node in nodes_to_consider_sorted_by_value:
            all_branches_to_open: list[BranchKey] = (
                self.opening_instructor.all_branches_to_open(node_to_open=node)
            )
            opening_instructions: OpeningInstructions[TNode] = (
                create_instructions_to_open_all_branches(
                    branches_to_play=all_branches_to_open, node_to_open=node
                )
            )
            opening_instructions_batch.merge(opening_instructions)

        self.current_depth_to_expand += 1
        return opening_instructions_batch

    def print_info(self) -> None:
        """
        Prints information about the Uniform Node selector.

        """
        print("Uniform")
