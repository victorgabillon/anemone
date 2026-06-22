"""Core structural helpers for expanding tree nodes and wiring descendants."""

from typing import TYPE_CHECKING, Any, cast

from valanga import BranchKey, State, StateModifications, StateTag

from anemone import nodes as node
from anemone import trees
from anemone.dynamics import SearchDynamics
from anemone.node_factory.base import (
    NodeFactory,
)
from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
)
from anemone.nodes.state_handles import MaterializedStateHandle
from anemone.tree_manager.branch_opening_service import BranchOpeningService
from anemone.tree_manager.opening_expansion_budget import OpeningExpansionBudget
from anemone.tree_manager.opening_expansion_executor import (
    OnePlyOpeningExpansionExecutor,
)
from anemone.tree_manager.tree_expander import (
    TreeExpansion,
    TreeExpansions,
)

if TYPE_CHECKING:
    from anemone.basics import TreeDepth


class DuplicateBranchOpenError(ValueError):
    """Raised when trying to open a branch that already has a child."""


def _duplicate_branch_open_error(
    *,
    parent_node: node.ITreeNode[Any],
    branch: BranchKey,
    child: node.ITreeNode[Any],
) -> DuplicateBranchOpenError:
    return DuplicateBranchOpenError(
        f"Cannot open branch {branch!r} from node {parent_node.id}: "
        f"branch already has child {child.id}"
    )


def _child_for_branch(
    parent_node: node.ITreeNode[Any],
    branch: BranchKey,
) -> node.ITreeNode[Any] | None:
    """Return a linked child, preferring the structural API when available."""
    child_for_branch = getattr(parent_node, "child_for_branch", None)
    if callable(child_for_branch):
        return cast("node.ITreeNode[Any] | None", child_for_branch(branch))
    return parent_node.branches_children.get(branch)


def _set_child_for_branch(
    parent_node: node.ITreeNode[Any],
    branch: BranchKey,
    child: node.ITreeNode[Any] | None,
) -> None:
    """Set a linked child, preferring the structural API when available."""
    set_child_for_branch = getattr(parent_node, "set_child_for_branch", None)
    if callable(set_child_for_branch):
        set_child_for_branch(branch, child)
        return
    parent_node.branches_children[branch] = child


def _discard_unopened_branch(
    parent_node: node.ITreeNode[Any],
    branch: BranchKey,
) -> None:
    """Discard an unopened branch, preferring the structural API when available."""
    discard_unopened_branch = getattr(parent_node, "discard_unopened_branch", None)
    if callable(discard_unopened_branch):
        discard_unopened_branch(branch)
        return
    parent_node.non_opened_branches.discard(branch)


def _raise_if_branch_already_opened(
    *,
    parent_node: node.ITreeNode[Any],
    branch: BranchKey,
) -> None:
    existing_child = _child_for_branch(parent_node, branch)
    if existing_child is not None:
        raise _duplicate_branch_open_error(
            parent_node=parent_node,
            branch=branch,
            child=existing_child,
        )


class TreeManager[
    FamilyT: node.ITreeNode[Any] = node.ITreeNode[Any],
]:
    """Manage structural tree expansion and descendant bookkeeping.

    This core manager is responsible for creating tree nodes and wiring them into
    the tree structure. Higher algorithm-aware layers consume the returned
    structural expansion records but do not push evaluation or propagation
    semantics down into this class.
    """

    node_factory: NodeFactory[FamilyT, Any]
    dynamics: SearchDynamics[Any, Any]

    def __init__(
        self,
        node_factory: NodeFactory[FamilyT, Any],
        dynamics: SearchDynamics[Any, Any],
    ) -> None:
        """Initialize the tree manager with a node factory and search dynamics."""
        self.node_factory = node_factory
        self.dynamics = dynamics

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
        self.assert_branch_not_opened(parent_node=parent_node, branch=branch)
        tree_depth = tree.node_depth(parent_node)
        transition = self.dynamics.step(parent_node.state, branch, depth=tree_depth)
        state = transition.next_state
        modifications = transition.modifications

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
        self.assert_branch_not_opened(parent_node=parent_node, branch=branch)
        # Creation of the child node. If the state already existed in another node, that node is returned as child_node.
        tree_depth: int = parent_node.tree_depth + 1
        state_tag: StateTag = state.tag

        need_creation_child_node: bool = tree.descendants.is_new_generation(
            tree_depth
        ) or not tree.descendants.contains_tag_at_depth(
            tree_depth=tree_depth,
            state_tag=state_tag,
        )

        tree_expansion: TreeExpansion[FamilyT]

        if need_creation_child_node:
            child_node: FamilyT
            child_node = self.node_factory.create(
                state_handle=MaterializedStateHandle(state_=state),
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
            child_node_existing = tree.descendants.node_at(
                tree_depth=tree_depth,
                state_tag=state_tag,
            )
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

        # Add it to opened branch links. Opening-status synchronization runs
        # after the whole instruction batch so stopping criteria can drop
        # requested branches without incorrectly marking the parent complete.
        _set_child_for_branch(parent_node, branch, tree_expansion.child_node)
        _discard_unopened_branch(parent_node, branch)
        tree.branch_count += 1  # counting branches

        return tree_expansion

    def expand_instructions(
        self,
        tree: trees.Tree[FamilyT],
        opening_instructions: OpeningInstructions[FamilyT],
        budget: OpeningExpansionBudget | None = None,
    ) -> TreeExpansions[FamilyT]:
        """Expand multiple branches and return purely structural expansion records.

        Args:
            tree: The tree object.
            opening_instructions: The opening instructions.
            budget: Runtime materialized branch-opening budget.

        Returns:
            The structural tree expansions that were performed.

        """
        executor = OnePlyOpeningExpansionExecutor(
            branch_opening_service=BranchOpeningService(tree_manager=self),
            dynamics=self.dynamics,
        )
        return executor.expand(
            tree=tree,
            opening_instructions=opening_instructions,
            budget=budget,
        )

    def assert_branch_not_opened(
        self,
        *,
        parent_node: node.ITreeNode[Any],
        branch: BranchKey,
    ) -> None:
        """Raise if ``branch`` already has a child from ``parent_node``."""
        _raise_if_branch_already_opened(parent_node=parent_node, branch=branch)

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
