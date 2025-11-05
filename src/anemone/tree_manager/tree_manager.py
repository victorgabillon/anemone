"""
This module contains the TreeManager class, which is responsible for managing a tree by opening new nodes and updating the values and indexes on the nodes.
"""

import typing

from valanga import BranchKey, State, StateTag, StateModifications

from anemone.basics import TreeDepth
import anemone.nodes as node
import anemone.trees as trees
from anemone.node_factory.node_factory import (
    TreeNodeFactory,
)
from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
)
from anemone.tree_manager.tree_expander import (
    TreeExpansion,
    TreeExpansions,
)

# todo should we use a discount? and discounted per round reward?
# todo maybe convenient to seperate this object into openner updater and dsiplayer
# todo have the reward with a discount
# DISCOUNT = 1/.99999
if typing.TYPE_CHECKING:
    import anemone.node_selector as node_sel


class TreeManager:
    """
    This class manages a tree by opening new nodes and updating the values and indexes on the nodes.
    """

    node_factory: TreeNodeFactory[node.ITreeNode]

    def __init__(self, node_factory: TreeNodeFactory[node.ITreeNode]) -> None:
        self.node_factory = node_factory

    def open_node_move(
        self,
        tree: trees.ValueTree,
        parent_node: node.ITreeNode,
        branch: BranchKey,
    ) -> TreeExpansion:
        """
        Opening a Node that contains a board following a move.

        Args:
            tree: The tree object.
            parent_node: The parent node that we want to expand.
            move: The move to play to expand the node.

        Returns:
            The tree expansion object.
        """
        # The parent board is copied, we only copy the stack (history of previous board) if the depth is smaller than 2
        # Having the stack information allows checking for draw by repetition.
        # To limit computation we limit copying it all the time. The resulting policy will only be aware of immediate
        # risk of draw by repetition
        copy_stack: bool = tree.node_depth(parent_node) < 2
        state: State = parent_node.state.copy(
            stack=copy_stack,
            deep_copy_legal_moves=False,  # trick to win time (the original legal moves is assume to not be changed as
            # moves are not supposed to be played anymore on that board and therefore this allows copy by reference
        )

        # The move is played. The board is now a new board
        modifications: StateModifications | None = state.step(
            branch_key=branch
        )


        return self.open_node(
            tree=tree,
            parent_node=parent_node,
            state=state,
            modifications=modifications,
            branch=branch,
        )

    def open_node(
        self,
        tree: trees.ValueTree,
        parent_node: node.ITreeNode,
        state: State,
        modifications: StateModifications| None,
        branch: BranchKey,
    ) -> TreeExpansion:
        """
        Opening a Node that contains a board given the modifications.
        Checks if the new node needs to be created or if the new_board already existed in the tree
         (was reached from a different serie of move)

        Args:
            tree: The tree object.
            parent_node: The parent node that we want to expand.
            board: The board object that is a move forward compared to the board in the parent node
            modifications: The board modifications.
            move: The move to play to expand the node.

        Returns:
            The tree expansion object.
        """

        # Creation of the child node. If the board already exited in another node, that node is returned as child_node.
        tree_depth: int = parent_node.tree_depth + 1
        state_tag: StateTag = state.tag

        child_node: node.ITreeNode
        need_creation_child_node: bool = (
            tree.descendants.is_new_generation(tree_depth)
            or state_tag not in tree.descendants.descendants_at_tree_depth[tree_depth]
        )
        if need_creation_child_node:
            child_node = self.node_factory.create(
                state=state,
                tree_depth=tree_depth,
                count=tree.nodes_count,
                branch_from_parent=branch,
                parent_node=parent_node,
            )
            tree.nodes_count += 1
            tree.descendants.add_descendant(
                child_node
            )  # add it to the list of descendants

        else:  # the node already exists
            child_node = tree.descendants[tree_depth][state_tag]
            child_node.add_parent(branch_key=branch, new_parent_node=parent_node)

        tree_expansion: TreeExpansion = TreeExpansion(
            child_node=child_node,
            parent_node=parent_node,
            board_modifications=modifications,
            creation_child_node=need_creation_child_node,
            branch_key=branch,
        )

        # add it to the list of opened move and out of the non-opened moves
        parent_node.branches_children[branch] = tree_expansion.child_node
        #   parent_node.tree_node.non_opened_legal_moves.remove(move)
        tree.move_count += 1  # counting moves

        return tree_expansion

    def open_instructions(
        self, tree: trees.ValueTree, opening_instructions: OpeningInstructions
    ) -> TreeExpansions:
        """
        Opening multiple nodes based on the opening instructions.

        Args:
            tree: The tree object.
            opening_instructions: The opening instructions.

        Returns:
            The tree expansions that have been performed.
        """

        # place to store the tree expansion logs generated by the openings
        tree_expansions: TreeExpansions = TreeExpansions()

        opening_instruction: node_sel.OpeningInstruction
        for opening_instruction in opening_instructions.values():
            # open
            tree_expansion: TreeExpansion = self.open_node_move(
                tree=tree,
                parent_node=opening_instruction.node_to_open,
                branch=opening_instruction.branch,
            )

            # concatenate the tree expansions
            tree_expansions.add(tree_expansion=tree_expansion)

        return tree_expansions

    def print_some_stats(
        self,
        tree: trees.ValueTree,
    ) -> None:
        """
        Print some statistics about the tree.

        Args:
            tree: The tree object.
        """
        print(
            "Tree stats: move_count",
            tree.move_count,
            " node_count",
            tree.descendants.get_count(),
        )
        sum_ = 0
        tree.descendants.print_stats()
        tree_depth: TreeDepth
        for tree_depth in tree.descendants:
            sum_ += len(tree.descendants[tree_depth])
            print("tree_depth", tree_depth, len(tree.descendants[tree_depth]), sum_)

    def test_count(
        self,
        tree: trees.ValueTree,
    ) -> None:
        """
        Test the count of nodes in the tree.

        Args:
            tree: The tree object.
        """
        assert tree.descendants.get_count() == tree.nodes_count

    def print_best_line(
        self,
        tree: trees.ValueTree,
    ) -> None:
        """
        Print the best line in the tree.

        Args:
            tree: The tree object.
        """
        raise Exception("should not be called no? Think about modifying...")
