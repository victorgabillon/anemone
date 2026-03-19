"""Provide Value-first NodeMinmaxEvaluation implementation for minimax search."""
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from math import log
from random import choice
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from valanga import (
    BranchKey,
    Color,
    TurnState,
)
from valanga.evaluations import Value

from anemone.dynamics import SearchDynamics
from anemone.node_evaluation.tree.adversarial.minmax_decision_ordering import (
    BranchSortValue,
    MinmaxDecisionOrderingState,
)
from anemone.node_evaluation.tree.node_tree_evaluation import NodeTreeEvaluationState
from anemone.nodes.itree_node import ITreeNode
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import AdversarialZeroSumObjective, Objective
from anemone.utils.logger import anemone_logger

if TYPE_CHECKING:
    from anemone.backup_policies.protocols import BackupPolicy
    from anemone.backup_policies.types import BackupResult


@runtime_checkable
# Class created to avoid circular import and defines what is seen and needed by the NodeMinmaxEvaluation class
class NodeWithValue(ITreeNode[TurnState], Protocol):
    """Represents a node with a value in a tree structure.

    Attributes:
        tree_evaluation (NodeMinmaxEvaluation): The minmax evaluation associated with the node.
        tree_node (TreeNode[Self]): The tree node associated with the node.

    Note: Uses Self to indicate that tree_node's children type should match the node itself.

    """

    tree_evaluation: "NodeMinmaxEvaluation"
    tree_node: TreeNode[Self, TurnState]


def make_minmax_decision_ordering_factory() -> MinmaxDecisionOrderingState:
    """Create the minimax child-ordering bookkeeping helper for one node."""
    return MinmaxDecisionOrderingState()


def make_default_objective() -> Objective[TurnState]:
    """Create the default objective preserving current adversarial semantics."""
    return AdversarialZeroSumObjective()


@dataclass(slots=True)
class NodeMinmaxEvaluation[
    NodeWithValueT: NodeWithValue = NodeWithValue,
    StateT: TurnState = TurnState,
](NodeTreeEvaluationState[NodeWithValueT, StateT]):
    r"""Value-first minimax evaluation attached to a tree node."""

    decision_ordering: MinmaxDecisionOrderingState = field(
        default_factory=make_minmax_decision_ordering_factory
    )

    # policy used to orchestrate backup behavior from updated children
    backup_policy: "BackupPolicy[Any] | None" = None

    # objective responsible for semantic interpretation of Value objects at this node
    objective: Objective[StateT] = field(default_factory=make_default_objective)

    @property
    def minmax_value(self) -> Value | None:
        """Return this family's storage-name alias for generic backed_up_value."""
        return self.backed_up_value

    @minmax_value.setter
    def minmax_value(self, value: Value | None) -> None:
        """Set this family's storage-name alias for generic backed_up_value."""
        self.backed_up_value = value

    @property
    def branches_sorted_by_value(self) -> dict[BranchKey, BranchSortValue]:
        """Return a dictionary containing the branches of the node sorted by their values.

        Returns:
            dict[BranchKey, BranchSortValue]: A dictionary where the keys are the branches in the node and
            the values are the corresponding sort values.

        """
        return self.decision_ordering.branches_sorted_by_value

    def is_over(self) -> bool:
        """Temporary compatibility alias for callers that still use legacy naming."""
        return self.is_terminal()

    def child_is_better_than_direct(
        self, child: Value, direct: Value, *, side_to_move: Color
    ) -> bool:
        """Determine if a child's value is better than the direct evaluation for the current node."""
        assert side_to_move == self.tree_node.state.turn
        return (
            self.objective.semantic_compare(
                child,
                direct,
                self.tree_node.state,
            )
            >= 0
        )

    def best_branch(self) -> BranchKey | None:
        """Return the best branch node based on the subjective value.

        Returns:
            The best branch based on the subjective value, or None if there are no branch open.

        """
        return self.decision_ordering.best_branch(
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=self._decision_semantic_compare,
        )

    def second_best_branch(self) -> BranchKey:
        """Return the second-best branch based on the subjective value.

        Returns:
            The second-best branch.

        """
        return self.decision_ordering.second_best_branch(
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=self._decision_semantic_compare,
        )

    def decision_ordered_branches(self) -> list[BranchKey]:
        """Return child branches ordered by current minimax decision semantics."""
        return self.decision_ordering.decision_ordered_branches(
            child_value_candidate_getter=self.child_value_candidate,
            semantic_compare=self._decision_semantic_compare,
        )

    def print_branches_sorted_by_value_and_exploration(
        self, dynamics: SearchDynamics[Any, Any]
    ) -> None:
        """Print the branch of the node sorted by their value and exploration.

        This method prints the branches of the node along with their subjective sort value.
        The branches are sorted based on their value and exploration.

        Args:
            dynamics (SearchDynamics): The dynamics used for labeling the edges in the visualization.

        Returns:
            None

        """
        branch_key: BranchKey
        anemone_logger.info(
            "here are the %s branches sorted by value: ",
            len(self.branches_sorted_by_value),
        )
        string_info: str = ""
        for branch_key, subjective_sort_value in self.branches_sorted_by_value.items():
            branch_name = dynamics.action_name(self.tree_node.state, branch_key)
            string_info += f" {branch_name} {subjective_sort_value[0]} $$ "
        anemone_logger.info(string_info)

    def branch_sort_value(self, branch_key: BranchKey) -> BranchSortValue:
        """Return the search-order tuple for one child branch.

        The first component is the objective's subjective sort value. The second
        component is a PV-length tie-break that consumes exact outcome metadata
        explicitly:

        - exact wins prefer shorter lines
        - exact losses prefer longer lines
        - draws / estimates / exact values without outcome metadata keep the
          default shorter-line preference
        """
        child = self.tree_node.branches_children[branch_key]
        assert child is not None
        child_value = self.child_value_candidate(branch_key)
        assert child_value is not None, (
            f"Cannot record sort value: child {branch_key} has no Value yet. "
            "Ensure children receive direct_value explicitly as a Value or "
            "minmax_value via backup before calling branch_sort_value()."
        )
        subjective_value_of_child = self.objective.evaluate_value(
            child_value,
            self.tree_node.state,
        )
        return (
            subjective_value_of_child,
            self._pv_length_tie_break(child=child, child_value=child_value),
            child.tree_node.id,
        )

    def _pv_length_tie_break(
        self,
        *,
        child: NodeWithValueT,
        child_value: Value,
    ) -> int:
        """Return the PV-length tie-break for one child.

        This must be based on the child branch's exact outcome metadata, not on
        whether the parent currently happens to expose ``over_event`` metadata.
        """
        pv_length = len(child.tree_evaluation.best_branch_sequence)
        over_event = child_value.over_event

        if over_event is not None and not over_event.is_draw():
            if over_event.is_winner(self.tree_node.state.turn):
                return pv_length
            return -pv_length

        return pv_length

    def are_equal_values[T](self, value_1: T, value_2: T) -> bool:
        """Check if two values are equal.

        Args:
            value_1 (T): The first value to compare.
            value_2 (T): The second value to compare.

        Returns:
            bool: True if the values are equal, False otherwise.

        """
        return value_1 == value_2

    def are_considered_equal_values[T](
        self, value_1: tuple[T, ...], value_2: tuple[T, ...]
    ) -> bool:
        """Check if two values are considered equal.

        Args:
            value_1 (tuple[T]): The first value to compare.
            value_2 (tuple[T]): The second value to compare.

        Returns:
            bool: True if the values are considered equal, False otherwise.

        """
        return value_1[:2] == value_2[:2]

    def are_almost_equal_values(self, value_1: float, value_2: float) -> bool:
        """Check if two float values are almost equal within a small epsilon.

        Args:
            value_1 (float): The first value to compare.
            value_2 (float): The second value to compare.

        Returns:
            bool: True if the values are almost equal, False otherwise.

        """
        epsilon = 0.01
        return value_1 > value_2 - epsilon and value_2 > value_1 - epsilon

    def update_branches_values(self, branches_to_consider: set[BranchKey]) -> None:
        """Update the values of the branches based on the given set of branches to consider.

        Args:
            branches_to_consider (set[BranchKey]): The set of branches to consider.

        Returns:
            None

        """
        self.decision_ordering.update_branches_values(
            branches_to_consider,
            branch_sort_value_getter=self.branch_sort_value,
        )

    def frontier_branches_in_order(self) -> list[BranchKey]:
        """Return frontier branches ordered by current child-preference semantics."""
        return self.branch_frontier.ordered_frontier_branches(
            (*self.branches_sorted_by_value, *self.tree_node.branches_children)
        )

    def _decision_semantic_compare(self, left: Value, right: Value) -> int:
        """Compare child values using current minimax decision semantics."""
        return self.objective.semantic_compare(left, right, self.tree_node.state)

    def assert_pv_invariants(self) -> None:
        """Assert core principal-variation invariants.

        Intended for tests and debugging guardrails.
        """
        best_branch_key = self.best_branch()

        if best_branch_key is None:
            assert not self.best_branch_sequence, (
                "PV must be empty when no best branch exists."
            )
            return

        if self.best_branch_sequence:
            assert self.best_branch_sequence[0] == best_branch_key, (
                "PV head must match best_branch()."
            )
            best_child = self.tree_node.branches_children.get(best_branch_key)
            assert best_child is not None, (
                "PV is non-empty but best child is missing from branches_children."
            )

        # NOTE: partial-expansion PV/value policy is owned by backup policies.

    def one_of_best_children_becomes_best_next_node(self) -> bool:
        """Triggered when the value of the current best branch does not match the best value.

        This method selects one of the best children nodes as the next best node based on a specific condition.
        It updates the `best_branch_sequence` attribute with the selected child node and its corresponding best branch sequence.

        Raises:
            AssertionError: If the number of best children is not equal to 1 when `how_equal_` is set to 'equal'.
            AssertionError: If the selected best child is not an instance of `NodeWithValue`.
            AssertionError: If the `best_node_sequence` attribute is empty after updating.

        """
        how_equal_: str = "equal"
        best_branches: list[BranchKey] = self.get_all_of_the_best_branches(
            how_equal=how_equal_
        )
        if how_equal_ == "equal":
            assert len(best_branches) == 1
        best_branch_key = choice(best_branches)
        best_child = self.tree_node.branches_children[best_branch_key]
        assert best_child is not None
        has_best_branch_seq_changed = self.set_best_branch_sequence(
            [
                best_branch_key,
                *best_child.tree_evaluation.best_branch_sequence,
            ]
        )
        assert self.best_branch_sequence
        return has_best_branch_seq_changed

    def backup_from_children(
        self,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> "BackupResult":
        """Delegate backup orchestration to the configured backup policy."""
        if self.backup_policy is None:
            from anemone.backup_policies.explicit_minimax import (  # pylint: disable=import-outside-toplevel
                ExplicitMinimaxBackupPolicy,
            )

            self.backup_policy = ExplicitMinimaxBackupPolicy()
        assert self.backup_policy is not None
        policy: BackupPolicy[Any] = self.backup_policy
        return policy.backup_from_children(
            node_eval=self,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )

    def print_best_line(self) -> None:
        """Print the best line from the current node to the leaf node.

        The best line is determined by following the sequence of child nodes with the highest values.
        Each child node is printed along with its corresponding branch and node ID.

        Returns:
            None

        """
        info_string: str = f"Best line from node {self.tree_node.id!s}: "
        minmax: Any = self
        for branch in self.best_branch_sequence:
            child = minmax.tree_node.branches_children[branch]
            assert child is not None
            info_string += f"{branch} ({child.tree_node.id!s}) "
            minmax = child.tree_evaluation
        anemone_logger.info(info_string)

    def my_logit(self, x: float) -> float:
        """Apply the logit function to the input value.

        Args:
            x (float): The input value.

        Returns:
            float: The result of applying the logit function to the input value.

        """
        y = min(max(x, 0.000000000000000000000001), 0.9999999999999999)
        return log(y / (1 - y)) * max(
            1, abs(x)
        )  # the * min(1,x) is a hack to prioritize game over

    def get_all_of_the_best_branches(
        self, how_equal: str | None = None
    ) -> list[BranchKey]:
        """Return a list of all the best branches based on the specified equality criteria.

        Args:
            how_equal (str | None): The equality criteria to determine the best branches.
                Possible values are 'equal', 'considered_equal', 'almost_equal', 'almost_equal_logistic'.
                Defaults to None.

        Returns:
            list[Ibranch]: A list of Ibranch representing the best branches.

        """
        best_branches: list[BranchKey] = []
        best_branch: BranchKey | None = self.best_branch()
        assert best_branch is not None
        best_value = self.branches_sorted_by_value[best_branch]
        branch_key: BranchKey
        for branch_key, branch_value in self.branches_sorted_by_value.items():
            if how_equal == "equal":
                if self.are_equal_values(branch_value, best_value):
                    best_branches.append(branch_key)
                    assert len(best_branches) == 1
            elif how_equal == "considered_equal":
                if self.are_considered_equal_values(branch_value, best_value):
                    best_branches.append(branch_key)
            elif how_equal == "almost_equal":
                if self.are_almost_equal_values(branch_value[0], best_value[0]):
                    best_branches.append(branch_key)
            elif how_equal == "almost_equal_logistic":
                best_value_logit = self.my_logit(best_value[0] * 0.5 + 0.5)
                child_value_logit = self.my_logit(branch_value[0] * 0.5 + 0.5)
                if self.are_almost_equal_values(child_value_logit, best_value_logit):
                    best_branches.append(branch_key)
        return best_branches
