"""
Sequool

This module contains the implementation of the Sequool node selector. The Sequool node selector is responsible for
choosing the best node to open in a move tree based on various selection strategies.

Classes:
- TreeDepthSelector: Protocol defining the interface for a half-move selector.
- StaticNotOpenedSelector: A node selector that considers the number of visits and selects half-moves based on zipf distribution.
- RandomAllSelector: A node selector that selects half-moves randomly.
- Sequool: The main class implementing the Sequool node selector.

Functions:
- consider_nodes_from_all_lesser_tree_depths_in_descendants: Consider all nodes in smaller half-moves than the picked half-move using the descendants object.
- consider_nodes_from_all_lesser_tree_depths_in_sub_stree: Consider all nodes in smaller half-moves than the picked half-move using tree traversal.
- consider_nodes_only_from_tree_depths_in_descendants: Consider only the nodes at the picked depth.
- get_best_node_from_candidates: Get the best node from a list of candidate nodes based on their exploration index data.
"""

import random
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


from anemone import trees
from anemone.basics import TreeDepth
from anemone.indices.node_indices.index_data import (
    MaxDepthDescendants,
)
from anemone.node_selector.notations_and_statics import (
    zipf_picks,
    zipf_picks_random,
)
from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
    create_instructions_to_open_all_branches,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.nodes.tree_traversal import (
    get_descendants_candidate_not_over,
)
from anemone.trees.descendants import Descendants

if typing.TYPE_CHECKING:
    import anemone.tree_manager as tree_man


class TreeDepthSelector[TNode: AlgorithmNode[Any] = AlgorithmNode[Any]](Protocol):
    """
    Protocol defining the interface for a half-move selector.
    """

    def update_from_expansions(
        self, latest_tree_expansions: "tree_man.TreeExpansions[TNode]"
    ) -> None:
        """
        Update the half-move selector with the latest tree expansions.

        Args:
            latest_tree_expansions: The latest tree expansions.

        Returns:
            None
        """

    def select_tree_depth(
        self, from_node: TNode, random_generator: random.Random
    ) -> TreeDepth:
        """
        Select the next half-move to consider based on the given node and random generator.

        Args:
            from_node: The current node.
            random_generator: The random generator.

        Returns:
            The selected half-move.
        """
        ...


@dataclass
class StaticNotOpenedSelector[TNode: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """
    A node selector that considers the number of visits and selects half-moves based on zipf distribution.
    """

    all_nodes_not_opened: Descendants[TNode]

    # counting the visits for each tree_depth
    count_visits: dict[TreeDepth, int] = field(
        default_factory=lambda: dict[TreeDepth, int]()
    )

    def update_from_expansions(
        self, latest_tree_expansions: "tree_man.TreeExpansions[TNode]"
    ) -> None:
        """
        Update the node selector with the latest tree expansions.

        Args:
            latest_tree_expansions: The latest tree expansions.

        Returns:
            None
        """

        # Update internal info with the latest tree expansions
        expansion: tree_man.TreeExpansion[TNode]
        for expansion in latest_tree_expansions:
            if expansion.creation_child_node:
                self.all_nodes_not_opened.add_descendant(expansion.child_node)

            # if a new tree_depth is being created then init the visits to 1
            # (0 would bug as it would automatically be selected with zipf computation)
            tree_depth: int = expansion.child_node.tree_depth
            if tree_depth not in self.count_visits:
                self.count_visits[tree_depth] = 1

    def select_tree_depth(
        self, from_node: TNode, random_generator: random.Random
    ) -> TreeDepth:
        """
        Select the next half-move to consider based on the given node and random generator.

        Args:
            from_node: The current node.
            random_generator: The random generator.

        Returns:
            The selected half-move.
        """

        filtered_count_visits: dict[int, int | float] = {
            hm: value
            for hm, value in self.count_visits.items()
            if hm in self.all_nodes_not_opened
        }

        # choose a half move based on zipf
        tree_depth_picked: int = zipf_picks(
            ranks_values=filtered_count_visits,
            random_generator=random_generator,
            shift=True,
            random_pick=False,
        )

        self.count_visits[tree_depth_picked] += 1

        return tree_depth_picked


type ConsiderNodesFromTreeDepths[TNode: AlgorithmNode[Any]] = Callable[
    [TreeDepth, TNode],
    list[TNode],
]


def consider_nodes_from_all_lesser_tree_depths_in_descendants[
    TNode: AlgorithmNode[Any]
](
    tree_depth_picked: TreeDepth,
    from_node: TNode,
    descendants: Descendants[TNode],
) -> list[TNode]:
    """
    Consider all the nodes that are in smaller half-moves than the picked half-move using the descendants object.

    Args:
        tree_depth_picked: The picked half-move.
        from_node: The current node.
        descendants: The descendants object.

    Returns:
        A list of nodes to consider.
    """

    nodes_to_consider: list[TNode] = []
    tree_depth: int
    # considering all half-move smaller than the half move picked
    for tree_depth in filter(lambda hm: hm < tree_depth_picked + 1, descendants):
        nodes_to_consider += list(descendants[tree_depth].values())

    return nodes_to_consider


def consider_nodes_from_all_lesser_tree_depths_in_sub_stree[TNode: AlgorithmNode[Any]](
    tree_depth_picked: TreeDepth,
    from_node: TNode,
) -> list[TNode]:
    """
    Consider all the nodes that are in smaller half-moves than the picked half-move using tree traversal.

    Args:
        tree_depth_picked: The picked half-move.
        from_node: The current node.

    Returns:
        A list of nodes to consider.
    """

    nodes_to_consider: list[TNode] = get_descendants_candidate_not_over(
        from_tree_node=from_node, max_depth=tree_depth_picked - from_node.tree_depth
    )
    return nodes_to_consider


def consider_nodes_only_from_tree_depths_in_descendants[TNode: AlgorithmNode[Any]](
    tree_depth_picked: TreeDepth,
    from_node: TNode,
    descendants: Descendants[TNode],
) -> list[TNode]:
    """
    Consider only the nodes at the picked depth.

    Args:
        tree_depth_picked: The picked half-move.
        from_node: The current node.
        descendants: The descendants object.

    Returns:
        A list of nodes to consider.
    """

    return list(descendants[tree_depth_picked].values())


@dataclass
class RandomAllSelector[TNode: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """
    A node selector that selects half-moves randomly.
    """

    def update_from_expansions(
        self, latest_tree_expansions: "tree_man.TreeExpansions[TNode]"
    ) -> None:
        """
        Update the node selector with the latest tree expansions.

        Args:
            latest_tree_expansions: The latest tree expansions.

        Returns:
            None
        """

    def select_tree_depth(
        self, from_node: TNode, random_generator: random.Random
    ) -> TreeDepth:
        """
        Select the next half-move to consider based on the given node and random generator.

        Args:
            from_node: The current node.
            random_generator: The random generator.

        Returns:
            The selected half-move.
        """

        tree_depth_picked: int
        # choose a half move based on zipf
        assert isinstance(from_node.exploration_index_data, MaxDepthDescendants)
        max_descendants_depth: int = (
            from_node.exploration_index_data.max_depth_descendants
        )
        if max_descendants_depth:
            depth_picked: int = zipf_picks_random(
                ordered_list_elements=list(range(1, max_descendants_depth + 1)),
                random_generator=random_generator,
            )
            tree_depth_picked = from_node.tree_depth + depth_picked
        else:
            tree_depth_picked = from_node.tree_depth
        return tree_depth_picked


def get_best_node_from_candidates[TNode: AlgorithmNode[Any]](
    nodes_to_consider: list[TNode],
) -> TNode:
    """
    Returns the best node from a list of candidate nodes based on their exploration index and half move.

    Args:
        nodes_to_consider (list[ITreeNode]): A list of candidate nodes to consider.

    Returns:
        AlgorithmNode: The best node from the list of candidates.
    """
    best_node: TNode = nodes_to_consider[0]
    assert best_node.exploration_index_data is not None
    best_value = (best_node.exploration_index_data.index, best_node.tree_depth)

    node: TNode
    for node in nodes_to_consider:
        assert node.exploration_index_data is not None
        if node.exploration_index_data.index is not None:
            assert best_node.exploration_index_data is not None
            if (
                best_node.exploration_index_data.index is None
                or (node.exploration_index_data.index, node.tree_depth) < best_value
            ):
                best_node = node
                best_value = (node.exploration_index_data.index, node.tree_depth)
    return best_node


@dataclass
class Sequool[TNode: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """
    The main class implementing the Sequool node selector.
    """

    opening_instructor: OpeningInstructor
    all_nodes_not_opened: Descendants[TNode]
    recursif: bool
    random_depth_pick: bool
    tree_depth_selector: TreeDepthSelector[TNode]
    random_generator: random.Random
    consider_nodes_from_tree_depths: ConsiderNodesFromTreeDepths[TNode]

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[TNode],
        latest_tree_expansions: "tree_man.TreeExpansions[TNode]",
    ) -> OpeningInstructions[TNode]:
        """
        Choose the best node to open in the move tree and return the opening instructions.

        Args:
            tree: The move tree.
            latest_tree_expansions: The latest tree expansions.

        Returns:
            The opening instructions.
        """

        self.tree_depth_selector.update_from_expansions(
            latest_tree_expansions=latest_tree_expansions
        )

        opening_instructions: OpeningInstructions[TNode] = (
            self.choose_node_and_move_to_open_recur(from_node=tree.root_node)
        )

        return opening_instructions

    def choose_node_and_move_to_open_recur(
        self, from_node: TNode
    ) -> OpeningInstructions[TNode]:
        """
        Recursively choose the best node to open in the move tree and return the opening instructions.

        Args:
            from_node: The current node.

        Returns:
            The opening instructions.
        """

        tree_depth_selected: TreeDepth = self.tree_depth_selector.select_tree_depth(
            from_node=from_node, random_generator=self.random_generator
        )

        nodes_to_consider: list[TNode] = self.consider_nodes_from_tree_depths(
            tree_depth_selected, from_node
        )

        best_node: TNode = get_best_node_from_candidates(
            nodes_to_consider=nodes_to_consider
        )

        if not self.recursif:
            self.all_nodes_not_opened.remove_descendant(best_node)

        if self.recursif and best_node.tree_node.all_branches_generated:
            return self.choose_node_and_move_to_open_recur(from_node=best_node)
        else:
            all_branches_to_open = self.opening_instructor.all_branches_to_open(
                node_to_open=best_node
            )
            opening_instructions: OpeningInstructions[TNode] = (
                create_instructions_to_open_all_branches(
                    branches_to_play=all_branches_to_open, node_to_open=best_node
                )
            )

            return opening_instructions
