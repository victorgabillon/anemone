"""Provide the NodeTreeEvaluation interface."""

from typing import TYPE_CHECKING, Any, Protocol

from valanga import (
    BranchKey,
    OverEvent,
    State,
    StateEvaluation,
)

from anemone.backup_policies.types import BackupResult
from anemone.dynamics import SearchDynamics
from anemone.values import Value

type BranchSortValue = tuple[float, int, int]


if TYPE_CHECKING:
    from anemone.nodes.itree_node import ITreeNode


class NodeTreeEvaluation[StateT: State = State](Protocol):
    """Interface for node tree evaluation.

    This is the evaluation of a node that is based both on a direct evaluation of the state within
    the NodeTreeEvaluation and its children.
    The direct evaluation is used to evaluate leaf nodes, while the children evaluations are used to propagate values up the tree.

    """

    # canonical direct evaluation value (Value-first API)
    direct_value: Value | None
    # canonical minmax value (Value-first API)
    minmax_value: Value | None

    # creating a base Over event that is set to None
    over_event: OverEvent

    # the list of branches that have not yet be found to be over
    # using atm a list instead of set as atm python set are not insertion ordered which adds randomness
    # and makes debug harder
    branches_not_over: list[BranchKey]

    branches_sorted_by_value_: dict[BranchKey, BranchSortValue]

    best_branch_sequence: list[BranchKey]

    def set_evaluation(self, evaluation: float) -> None:
        """Set the evaluation from the state evaluator.

        Args:
            evaluation (float): The evaluation value to be set.

        Returns:
            None

        """
        ...

    def is_over(self) -> bool:
        """Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.

        """
        ...

    def is_terminal_candidate(self) -> bool:
        """Return whether the canonical Value candidate is terminal/forced with over metadata."""
        ...

    def dot_description(self) -> str:
        """Return a string representation of the node's description in DOT format.

        The description includes canonical evaluation information,
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

    def backup_from_children(
        self,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Run backup policy from updated children and return changed-state flags."""
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

    def get_score(self) -> float:
        """Return the canonical scalar score for this node evaluation."""
        ...

    def get_value_candidate(self) -> Value | None:
        """Return minmax when available, else direct Value, or ``None``."""
        ...

    def require_value_candidate(self) -> Value:
        """Return minmax when available, else direct Value, raising if unavailable."""
        ...

    def get_value(self) -> Value:
        """Return the canonical Value used by minimax and ordering logic."""
        ...

    def sync_over_from_values(self) -> None:
        """Synchronize ``over_event`` from canonical terminal Value metadata."""
        ...

    def best_branch(self) -> BranchKey | None:
        """Return the current best branch key."""
        ...

    def second_best_branch(self) -> BranchKey:
        """Return the second-best branch key."""
        ...

    def print_branches_sorted_by_value(
        self, dynamics: SearchDynamics[Any, Any]
    ) -> None:
        """Print branches sorted by value."""
        ...

    def print_branches_sorted_by_value_and_exploration(
        self, dynamics: SearchDynamics[Any, Any]
    ) -> None:
        """Print branches sorted by value and exploration metrics."""
        ...

    def get_all_of_the_best_branches(
        self, how_equal: str | None = None
    ) -> list[BranchKey]:
        """Return all best branches according to an equality rule."""
        ...

    def sort_branches_not_over(self) -> list[BranchKey]:
        """Return branches not over, sorted by evaluation."""
        ...
