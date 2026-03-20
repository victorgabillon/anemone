"""Provide Value-first NodeMinmaxEvaluation implementation for minimax search."""
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Self, runtime_checkable

from valanga import (
    BranchKey,
    Color,
    TurnState,
)
from valanga.evaluations import Value

from anemone.dynamics import SearchDynamics
from anemone.node_evaluation.tree.decision_ordering import (
    BranchOrderingKey,
)
from anemone.node_evaluation.tree.node_tree_evaluation import (
    BackupPolicyFactory,
    NodeTreeEvaluationState,
    TreeEvaluationChild,
)
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import AdversarialZeroSumObjective, Objective
from anemone.utils.logger import anemone_logger

if TYPE_CHECKING:
    from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy


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

    _default_backup_policy_factory: ClassVar[BackupPolicyFactory | None] = (
        make_default_backup_policy
    )

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

    def _ensure_decision_ordering_ready(self) -> None:
        """Preserve minimax's incremental ordering-update policy."""

    def _decision_semantic_compare(self, left: Value, right: Value) -> int:
        """Compare child values using current minimax decision semantics."""
        return self.objective.semantic_compare(left, right, self.tree_node.state)

    def _branch_values_are_equal(
        self,
        *,
        branch: BranchKey,
        best_branch: BranchKey,
    ) -> bool:
        """Return whether branches match on exact Value and minimax PV tie-break."""
        branch_value = self.child_value_candidate(branch)
        best_value = self.child_value_candidate(best_branch)
        assert branch_value is not None
        assert best_value is not None

        branch_child = self.tree_node.branches_children[branch]
        best_child = self.tree_node.branches_children[best_branch]
        assert branch_child is not None
        assert best_child is not None

        return branch_value == best_value and self._pv_length_tie_break(
            child=branch_child,
            child_value=branch_value,
        ) == self._pv_length_tie_break(
            child=best_child,
            child_value=best_value,
        )

    def _branch_values_are_considered_equal(
        self,
        *,
        branch: BranchKey,
        best_branch: BranchKey,
    ) -> bool:
        """Return whether two branches tie under minimax decision semantics."""
        branch_value = self.child_value_candidate(branch)
        best_value = self.child_value_candidate(best_branch)
        assert branch_value is not None
        assert best_value is not None
        return self.objective.semantic_compare(
            branch_value,
            best_value,
            self.tree_node.state,
        ) == 0

    def one_of_best_children_becomes_best_next_node(self) -> bool:
        """Refresh the PV head from the currently selected deterministic best child."""
        best_branch_key = self.best_branch()
        assert best_branch_key is not None
        has_best_branch_seq_changed = self.set_best_branch_sequence(
            self.best_branch_line_from_child(best_branch_key)
        )
        assert self.best_branch_sequence
        return has_best_branch_seq_changed
