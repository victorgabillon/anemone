"""Provide Value-first NodeMinmaxEvaluation implementation for minimax search."""
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from random import choice
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from valanga import (
    BranchKey,
    Color,
    TurnState,
)
from valanga.evaluations import Value

from anemone.dynamics import SearchDynamics
from anemone.node_evaluation.tree.decision_ordering import (
    BranchOrderingKey,
    DecisionOrderingState,
)
from anemone.node_evaluation.tree.node_tree_evaluation import (
    BestBranchEquivalenceMode,
    NodeTreeEvaluationState,
    TreeEvaluationChild,
)
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import AdversarialZeroSumObjective, Objective
from anemone.utils.logger import anemone_logger

if TYPE_CHECKING:
    from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy
    from anemone.backup_policies.protocols import BackupPolicy
    from anemone.backup_policies.types import BackupResult


@runtime_checkable
# Class created to avoid circular import and defines what is seen and needed by the NodeMinmaxEvaluation class
class NodeWithValue(TreeEvaluationChild[TurnState], Protocol):
    """Represents a node with a value in a tree structure.

    Attributes:
        tree_evaluation (NodeMinmaxEvaluation): The minmax evaluation associated with the node.
        tree_node (TreeNode[Self]): The tree node associated with the node.

    Note: Uses Self to indicate that tree_node's children type should match the node itself.

    """

    @property
    def tree_evaluation(self) -> "NodeMinmaxEvaluation":
        """Return the minimax evaluation associated with this node."""
        ...

    tree_node: TreeNode[Self, TurnState]

def make_default_objective() -> Objective[TurnState]:
    """Create the default objective preserving current adversarial semantics."""
    return AdversarialZeroSumObjective()


def make_default_backup_policy() -> "ExplicitMinimaxBackupPolicy":
    """Create the default explicit minimax backup policy."""
    from anemone.backup_policies.explicit_minimax import (  # pylint: disable=import-outside-toplevel
        ExplicitMinimaxBackupPolicy,
    )

    return ExplicitMinimaxBackupPolicy()


@dataclass(slots=True)
class NodeMinmaxEvaluation[
    NodeWithValueT: NodeWithValue = NodeWithValue,
    StateT: TurnState = TurnState,
](NodeTreeEvaluationState[NodeWithValueT, StateT]):
    r"""Value-first minimax evaluation attached to a tree node."""

    decision_ordering: DecisionOrderingState = field(default_factory=DecisionOrderingState)

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
    def branch_ordering_keys(self) -> dict[BranchKey, BranchOrderingKey]:
        """Return the cached generic branch-ordering keys for this node."""
        return self.decision_ordering.branch_ordering_keys

    @property
    def branches_sorted_by_value(self) -> dict[BranchKey, BranchOrderingKey]:
        """Compatibility alias for callers that still use the legacy name."""
        return self.branch_ordering_keys

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

    def branch_sort_value(self, branch_key: BranchKey) -> BranchOrderingKey:
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

    def update_branches_values(self, branches_to_consider: set[BranchKey]) -> None:
        """Update the values of the branches based on the given set of branches to consider.

        Args:
            branches_to_consider (set[BranchKey]): The set of branches to consider.

        Returns:
            None

        """
        self.decision_ordering.update_branch_ordering_keys(
            branches_to_consider,
            branch_ordering_key_getter=self.branch_sort_value,
        )

    def _ordered_candidate_branches_for_frontier(self) -> tuple[BranchKey, ...]:
        """Return frontier candidates in cached minimax search order."""
        return (*self.branch_ordering_keys, *self.tree_node.branches_children)

    def _ordered_candidate_branches_for_best_equivalence(
        self,
    ) -> tuple[BranchKey, ...]:
        """Return candidate branches in cached minimax search order."""
        return tuple(self.branch_ordering_keys)

    def _branch_ordering_key(self, branch: BranchKey) -> BranchOrderingKey:
        """Return the cached branch-ordering key for one branch."""
        return self.branch_ordering_keys[branch]

    def _branch_values_are_equal(
        self,
        *,
        branch: BranchKey,
        best_branch: BranchKey,
    ) -> bool:
        """Return whether two minimax branch ordering keys are exactly equal."""
        return self._branch_ordering_key(branch) == self._branch_ordering_key(
            best_branch
        )

    def _decision_semantic_compare(self, left: Value, right: Value) -> int:
        """Compare child values using current minimax decision semantics."""
        return self.objective.semantic_compare(left, right, self.tree_node.state)

    def one_of_best_children_becomes_best_next_node(self) -> bool:
        """Triggered when the value of the current best branch does not match the best value.

        This method selects one of the best children nodes as the next best node based on a specific condition.
        It updates the `best_branch_sequence` attribute with the selected child node and its corresponding best branch sequence.

        Raises:
            AssertionError: If the number of best children is not equal to 1 when `how_equal_` is set to 'equal'.
            AssertionError: If the selected best child is not an instance of `NodeWithValue`.
            AssertionError: If the `best_node_sequence` attribute is empty after updating.

        """
        mode = BestBranchEquivalenceMode.EQUAL
        best_branches: list[BranchKey] = self.best_equivalent_branches(mode=mode)
        assert len(best_branches) == 1
        best_branch_key = choice(best_branches)
        has_best_branch_seq_changed = self.set_best_branch_sequence(
            self.best_branch_line_from_child(best_branch_key)
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
            self.backup_policy = make_default_backup_policy()
        assert self.backup_policy is not None
        policy: BackupPolicy[Any] = self.backup_policy
        return policy.backup_from_children(
            node_eval=self,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )
