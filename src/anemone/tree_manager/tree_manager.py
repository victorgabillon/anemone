"""Module containing the TreeManager class for managing tree expansions."""

from typing import TYPE_CHECKING, Any

from valanga import BranchKey, State, StateModifications, StateTag

from anemone import nodes as node
from anemone import trees
from anemone.node_factory.base import (
    NodeFactory,
)
from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
)
from anemone.state_transition import StateTransition
from anemone.tree_manager.tree_expander import (
    TreeExpansion,
    TreeExpansions,
    record_tree_expansion,
)

if TYPE_CHECKING:
    from anemone.basics import TreeDepth


class TreeManager[
    FamilyT: node.ITreeNode[Any] = node.ITreeNode[Any],
]:
    """Manage a tree by opening new nodes and tracking expansions.

    This core manager is responsible for creating tree nodes and wiring them into
    the tree structure.
    """

    node_factory: NodeFactory[FamilyT]
    transition: StateTransition[State]

    def __init__(
        self,
        node_factory: NodeFactory[FamilyT],
        transition: StateTransition[State],
    ) -> None:
        """Initialize the tree manager with a node factory and transition."""
        self.node_factory = node_factory
        self.transition = transition

    def open_tree_expansion_from_branch(
        self,
        tree: trees.Tree[FamilyT],
        parent_node: FamilyT,
        branch: BranchKey,
    ) -> TreeExpansion[FamilyT]:
        """Open a child node by applying a branch to the parent state.

        Args:
            tree: The tree object.
            parent_node: The parent node that we want to expand.
            branch: The branch key to apply.

        Returns:
            The tree expansion object.

        """
        # The parent state is copied; we only copy the stack (history of previous states) if the depth is smaller than 2.
        # Having the stack information allows checking for draw by repetition.
        # To limit computation we limit copying it all the time. The resulting policy will only be aware of immediate
        # risk of draw by repetition
        copy_stack: bool = tree.node_depth(parent_node) < 2
        parent_state: State = parent_node.state
        state: State = self.transition.copy_for_expansion(
            parent_state,
            copy_stack=copy_stack,
        )

        # The branch is applied. The state is now advanced.
        state, modifications = self.transition.step(state, branch_key=branch)

        return self.open_tree_expansion_from_state(
            tree=tree,
            parent_node=parent_node,
            state=state,
            modifications=modifications,
            branch=branch,
        )

    def open_tree_expansion_from_state(
        self,
        tree: trees.Tree[FamilyT],
        parent_node: FamilyT,
        state: State,
        modifications: StateModifications | None,
        branch: BranchKey,
    ) -> TreeExpansion[FamilyT]:
        """Open or reuse a child node for a specific state.

        Checks whether a new node must be created or whether the state already exists
        in the tree (reached from another branch sequence).

        Args:
            tree: The tree object.
            parent_node: The parent node that we want to expand.
            state: The state reached from the parent node.
            modifications: The state modifications produced by the branch.
            branch: The branch key applied to reach the state.

        Returns:
            The tree expansion object.

        """
        # Creation of the child node. If the state already existed in another node, that node is returned as child_node.
        tree_depth: int = parent_node.tree_depth + 1
        state_tag: StateTag = state.tag

        need_creation_child_node: bool = (
            tree.descendants.is_new_generation(tree_depth)
            or state_tag not in tree.descendants.descendants_at_tree_depth[tree_depth]
        )

        tree_expansion: TreeExpansion[FamilyT]

        if need_creation_child_node:
            child_node: FamilyT
            child_node = self.node_factory.create(
                state=state,
                tree_depth=tree_depth,
                count=tree.nodes_count,
                branch_from_parent=branch,
                parent_node=parent_node,
                modifications=modifications,
            )

            tree_expansion = TreeExpansion(
                child_node=child_node,
                parent_node=parent_node,
                state_modifications=modifications,
                creation_child_node=need_creation_child_node,
                branch_key=branch,
            )

        else:  # the node already exists
            child_node_existing: FamilyT
            child_node_existing = tree.descendants[tree_depth][state_tag]
            child_node_existing.add_parent(
                branch_key=branch, new_parent_node=parent_node
            )

            tree_expansion = TreeExpansion(
                child_node=child_node_existing,
                parent_node=parent_node,
                state_modifications=modifications,
                creation_child_node=need_creation_child_node,
                branch_key=branch,
            )

        # add it to the list of opened branches and out of the non-opened branches
        parent_node.branches_children[branch] = tree_expansion.child_node
        tree.branch_count += 1  # counting branches

        return tree_expansion

    def open_instructions(
        self,
        tree: trees.Tree[FamilyT],
        opening_instructions: OpeningInstructions[FamilyT],
    ) -> TreeExpansions[FamilyT]:
        """Open multiple nodes based on the opening instructions.

        Args:
            tree: The tree object.
            opening_instructions: The opening instructions.

        Returns:
            The tree expansions that have been performed.

        """
        # place to store the tree expansion logs generated by the openings
        tree_expansions: TreeExpansions[FamilyT] = TreeExpansions()

        opening_instruction: OpeningInstruction[FamilyT]
        for opening_instruction in opening_instructions.values():
            # open
            tree_expansion: TreeExpansion[FamilyT] = (
                self.open_tree_expansion_from_branch(
                    tree=tree,
                    parent_node=opening_instruction.node_to_open,
                    branch=opening_instruction.branch,
                )
            )

            record_tree_expansion(
                tree=tree,
                tree_expansions=tree_expansions,
                tree_expansion=tree_expansion,
            )

        return tree_expansions

    def print_some_stats(
        self,
        tree: trees.Tree[FamilyT],
    ) -> None:
        """Print some statistics about the tree.

        Args:
            tree: The tree object.

        """
        print(
            "Tree stats: branch_count",
            tree.branch_count,
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
        tree: trees.Tree[FamilyT],
    ) -> None:
        """Test the count of nodes in the tree.

        Args:
            tree: The tree object.

        """
        assert tree.descendants.get_count() == tree.nodes_count

    def print_best_line(
        self,
        tree: trees.Tree[FamilyT],
    ) -> None:
        """Print the best line in the tree.

        Args:
            tree: The tree object.

        """
        raise NotImplementedError(
            "print_best_line should not be called; override or modify this behavior"
        )
