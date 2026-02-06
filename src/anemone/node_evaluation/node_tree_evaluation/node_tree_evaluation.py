"""Provide the NodeTreeEvaluation interface."""


from typing import TYPE_CHECKING, Protocol, Self

from valanga import (
    BranchKey,
    OverEvent,
    State,
    StateEvaluation,
)

type BranchSortValue = tuple[float, int, int]


if TYPE_CHECKING:
    from anemone.nodes.itree_node import ITreeNode


class NodeTreeEvaluation[StateT: State = State](Protocol):
    """Interface for Node Tree Evaluation
    This is the evaluation of a node that is based both on a direct evaluation of the state within and of the NodeTreeEvaluation
    and its children.
    The direct evaluation is used to evaluate leaf nodes, while the children evaluations are used to propagate values up the tree.

    """

    # absolute value wrt to white player as estimated by a state evaluator
    value_white_direct_evaluation: float | None = None

    # creating a base Over event that is set to None
    over_event: OverEvent

    # the list of branches that have not yet be found to be over
    # using atm a list instead of set as atm python set are not insertion ordered which adds randomness
    # and makes debug harder
    branches_not_over: list[BranchKey]

    branches_sorted_by_value_: dict[BranchKey, BranchSortValue]

    best_branch_sequence: list[BranchKey]

    # absolute value wrt to white player as computed from the value_white_* of the descendants
    # of this node (self) by a minmax procedure.
    value_white_minmax: float | None = None

    def set_evaluation(self, evaluation: float) -> None:
        """Set the evaluation from the state evaluator.

        Args:
            evaluation (float): The evaluation value to be set.

        Returns:
            None

        """
        ...

    def is_over(self) -> bool:
        """Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.

        """
        ...

    def dot_description(self) -> str:
        """Returns a string representation of the node's description in DOT format.

        The description includes the values of `value_white_minmax` and `value_white_evaluator`,
        as well as the best branch sequence and the over event tag.

        Returns:
            A string representation of the node's description in DOT format.

        """
        ...

    def update_best_branch_sequence(
        self, branches_with_updated_best_branch_seq: set[BranchKey]
    ) -> bool:
        """Update the best branch sequence from updated branches."""
        ...

    def minmax_value_update_from_children(
        self, branches_with_updated_value: set[BranchKey]
    ) -> tuple[bool, bool]:
        """Update minmax value from children and return update flags."""
        ...

    def update_over(self, branches_with_updated_over: set[BranchKey]) -> bool:
        """Update terminal state based on updated branches."""
        ...

    def evaluate(self) -> StateEvaluation:
        """Return a state evaluation for this node."""
        ...

    def description_tree_visualizer_branch(self, child: "ITreeNode[StateT]") -> str:
        """Return a visualization label for a child branch."""
        ...

    def print_best_line(self) -> None:
        """Print the current best line."""
        ...

    def get_value_white(self) -> float:
        """Return the current white evaluation value."""
        ...

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch key."""
        ...

    def second_best_branch(self) -> BranchKey:
        """Return the second-best branch key."""
        ...

    def print_branches_sorted_by_value(self) -> None:
        """Print branches sorted by value."""
        ...

    def print_branches_sorted_by_value_and_exploration(self) -> None:
        """Print branches sorted by value and exploration metrics."""
        ...

    def get_all_of_the_best_branches(
        self, how_equal: str | None = None
    ) -> list[BranchKey]:
        """Return all best branches according to an equality rule."""
        ...

    def subjective_value_of(self, another_node_eval: Self) -> float:
        """Return this node's value relative to another evaluation."""
        ...

    def sort_branches_not_over(self) -> list[BranchKey]:
        """Return branches not over, sorted by evaluation."""
        ...
