"""Define stopping criteria for a branch selector in a game tree.

The stopping criteria determine when the selector should stop exploring the game tree and make a decision.

The module includes the following classes:

- StoppingCriterion: The general stopping criterion class.
- TreeBranchLimit: A stopping criterion based on a tree branch limit.
- DepthLimit: A stopping criterion based on a depth limit.

It also includes helper classes and functions for creating and managing stopping criteria.
"""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from anemone import node_selector as node_sel
from anemone import trees
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode


@runtime_checkable
class DepthToExpendP(Protocol):
    """Protocol for objects that provide the current depth to expand.

    This protocol defines a single method `get_current_depth_to_expand` that should be implemented by classes
    that want to provide the current depth to expand.

    Attributes:
        None

    Methods:
        get_current_depth_to_expand: Returns the current depth to expand as an integer.

    Examples:
        >>> class MyDepthToExpend(DepthToExpendP):
        ...     def get_current_depth_to_expand(self) -> int:
        ...         return 5
        ...
        >>> obj = MyDepthToExpend()
        >>> obj.get_current_depth_to_expand()
        5

    """

    def get_current_depth_to_expand(self) -> int:
        """Return the current depth to expand as an integer.

        Returns:
            The current depth to expand.

        Raises:
            None

        """
        ...


class StoppingCriterionTypes(StrEnum):
    """Enum class representing different types of stopping criteria for tree value calculation."""

    DEPTH_LIMIT = "depth_limit"
    TREE_BRANCH_LIMIT = "tree_branch_limit"


@dataclass
class StoppingCriterionArgs:
    """Represents the arguments for a stopping criterion.

    Attributes:
        type (StoppingCriterionTypes): The type of stopping criterion.

    """

    type: StoppingCriterionTypes


class ProgressMonitorP[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](Protocol):
    """The general stopping criterion Protocol."""

    def should_we_continue(self, tree: trees.Tree[NodeT]) -> bool:
        """Asking should we continue.

        Returns:
            boolean of should we continue

        """
        ...

    def respectful_opening_instructions(
        self,
        opening_instructions: node_sel.OpeningInstructions[NodeT],
        tree: trees.Tree[NodeT],
    ) -> node_sel.OpeningInstructions[NodeT]:
        """Ensure the opening request does not exceed the stopping criterion."""
        ...

    def get_string_of_progress(self, tree: trees.Tree[NodeT]) -> str:
        """Return a string representation of the progress made by the stopping criterion.

        Args:
            tree (Tree): The tree being explored.

        Returns:
            str: A string representation of the progress.

        """
        ...

    def get_percent_of_progress(
        self,
        tree: trees.Tree[NodeT],
    ) -> str:
        """Return a human-readable percent progress string."""
        ...


class ProgressMonitor[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """The general stopping criterion base class."""

    def should_we_continue(self, tree: trees.Tree[NodeT]) -> bool:
        """Asking should we continue.

        Returns:
            boolean of should we continue

        """
        return not tree.root_node.is_over()

    def respectful_opening_instructions(
        self,
        opening_instructions: node_sel.OpeningInstructions[NodeT],
        tree: trees.Tree[NodeT],
    ) -> node_sel.OpeningInstructions[NodeT]:
        """Ensure the opening request does not exceed the stopping criterion."""
        _ = tree
        return opening_instructions

    def get_string_of_progress(self, _tree: trees.Tree[NodeT]) -> str:
        """Return a string representation of the progress made by the stopping criterion.

        Args:
            tree (Tree): The tree being explored.

        Returns:
            str: A string representation of the progress.

        """
        return ""

    @abstractmethod
    def get_percent_of_progress(self, tree: trees.Tree[NodeT]) -> int:
        """Return a numeric progress percentage for this monitor."""
        ...

    def notify_percent_progress(
        self,
        tree: trees.Tree[NodeT],
        notify_percent_function: Callable[[int], None] | None,
    ) -> None:
        """Notify a callback with the current progress percentage."""
        percent_progress: int = self.get_percent_of_progress(tree=tree)

        if notify_percent_function is not None:
            notify_percent_function(percent_progress)


@dataclass
class TreeBranchLimitArgs:
    """Arguments for the tree branch limit stopping criterion."""

    type: Literal[StoppingCriterionTypes.TREE_BRANCH_LIMIT]
    tree_branch_limit: int


class TreeBranchLimit[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](
    ProgressMonitor[NodeT]
):
    """The stopping criterion based on a tree branch limit."""

    tree_branch_limit: int

    def __init__(self, tree_branch_limit: int) -> None:
        """Initialize the monitor with a branch-count limit."""
        self.tree_branch_limit = tree_branch_limit

    def should_we_continue(self, tree: trees.Tree[NodeT]) -> bool:
        """Return True while within the branch-count budget."""
        continue_base: bool = super().should_we_continue(tree=tree)

        should_we: bool
        should_we = continue_base and tree.branch_count < self.tree_branch_limit
        return should_we

    def respectful_opening_instructions(
        self,
        opening_instructions: node_sel.OpeningInstructions[NodeT],
        tree: trees.Tree[NodeT],
    ) -> node_sel.OpeningInstructions[NodeT]:
        """Ensure the opening request does not exceed the stopping criterion."""
        opening_instructions_subset: node_sel.OpeningInstructions[NodeT] = (
            node_sel.OpeningInstructions()
        )
        opening_instructions.pop_items(
            popped=opening_instructions_subset,
            how_many=self.tree_branch_limit - tree.branch_count,
        )
        return opening_instructions_subset

    def get_string_of_progress(self, tree: trees.Tree[NodeT]) -> str:
        """Compute the string that display the progress in the terminal.

        Returns:
            a string that display the progress in the terminal

        """
        return (
            f"========= tree branch counting: {tree.branch_count} out of {self.tree_branch_limit}"
            f" |  {tree.branch_count / self.tree_branch_limit:.0%}"
        )

    def get_percent_of_progress(
        self,
        tree: trees.Tree[NodeT],
    ) -> int:
        """Return progress percentage based on branch count."""
        percent: int = int(tree.branch_count / self.tree_branch_limit * 100)
        return percent


@dataclass
class DepthLimitArgs:
    """Arguments for the depth limit stopping criterion.

    Attributes:
        depth_limit (int): The maximum depth allowed for the search.

    """

    type: Literal[StoppingCriterionTypes.DEPTH_LIMIT]
    depth_limit: int


class DepthLimit[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](
    ProgressMonitor[NodeT]
):
    """The stopping criterion based on a depth limit."""

    depth_limit: int
    node_selector: DepthToExpendP

    def __init__(self, depth_limit: int, node_selector: DepthToExpendP) -> None:
        """Initialize a StoppingCriterion object.

        Args:
            depth_limit (int): The maximum depth to search in the tree.
            node_selector (DepthToExpendP): The node selector used to determine which nodes to expand.

        Returns:
            None

        """
        self.depth_limit = depth_limit
        self.node_selector = node_selector

    def should_we_continue(self, tree: trees.Tree[NodeT]) -> bool:
        """Determine whether the search should continue expanding nodes in the tree.

        Args:
            tree (Tree): The tree containing the nodes and their evaluations.

        Returns:
            bool: True if the search should continue, False otherwise.

        """
        continue_base = super().should_we_continue(tree=tree)
        if not continue_base:
            return continue_base
        return self.node_selector.get_current_depth_to_expand() < self.depth_limit

    def get_string_of_progress(self, tree: trees.Tree[NodeT]) -> str:
        """Compute the string that display the progress in the terminal.

        Returns:
            a string that display the progress in the terminal

        """
        return (
            "========= tree branch counting: "
            + str(tree.branch_count)
            + " | Depth: "
            + str(self.node_selector.get_current_depth_to_expand())
            + " out of "
            + str(self.depth_limit)
        )

    def get_percent_of_progress(
        self,
        tree: trees.Tree[NodeT],
    ) -> int:
        """Return progress percentage based on current depth."""
        # TODO: this percent is not precise
        percent: int = int(
            self.node_selector.get_current_depth_to_expand() / self.depth_limit * 100
        )
        return percent


AllStoppingCriterionArgs = TreeBranchLimitArgs | DepthLimitArgs


class UnknownStoppingCriterionError(ValueError):
    """Raised when a stopping criterion type is not recognized."""

    def __init__(self, criterion_type: StoppingCriterionTypes) -> None:
        """Initialize the error with the unsupported criterion type."""
        super().__init__(
            f"stopping criterion builder: can not find {criterion_type} in file {__name__}"
        )


def create_stopping_criterion[NodeT: AlgorithmNode[Any]](
    args: AllStoppingCriterionArgs,
    node_selector: node_sel.NodeSelector[NodeT],
) -> ProgressMonitor[NodeT]:
    """Create the stopping criterion.

    Args:
        args (AllStoppingCriterionArgs): Configuration for the stopping criterion.
        node_selector (node_sel.NodeSelector[NodeT]): Node selector used by the criterion.

    Returns:
        ProgressMonitor[NodeT]: The constructed stopping criterion.

    """
    stopping_criterion: ProgressMonitor[NodeT]

    match args.type:
        case StoppingCriterionTypes.DEPTH_LIMIT:
            assert isinstance(node_selector, DepthToExpendP)
            assert isinstance(args, DepthLimitArgs)
            stopping_criterion = DepthLimit(
                depth_limit=args.depth_limit, node_selector=node_selector
            )
        case StoppingCriterionTypes.TREE_BRANCH_LIMIT:
            assert isinstance(args, TreeBranchLimitArgs)

            stopping_criterion = TreeBranchLimit(
                tree_branch_limit=args.tree_branch_limit
            )
        case _:
            raise UnknownStoppingCriterionError(args.type)

    return stopping_criterion
