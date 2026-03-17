"""Provide Value-first NodeMinmaxEvaluation implementation for minimax search."""
# pylint: disable=duplicate-code

# TODO: maybe further split values from over?

from dataclasses import dataclass, field
from functools import cmp_to_key
from math import log
from random import choice
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from valanga import (
    BranchKey,
    Color,
    OverEvent,
    TurnState,
)
from valanga.evaluations import Value

from anemone.dynamics import SearchDynamics
from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.common.branch_frontier import BranchFrontierState
from anemone.nodes.itree_node import ITreeNode
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import AdversarialZeroSumObjective, Objective
from anemone.utils.logger import anemone_logger
from anemone.utils.my_value_sorted_dict import sort_dic
from anemone.values import (
    DEFAULT_EVALUATION_ORDERING,
    EvaluationOrdering,
)

if TYPE_CHECKING:
    from anemone.backup_policies.protocols import BackupPolicy
    from anemone.backup_policies.types import BackupResult

type BranchSortValue = tuple[float, int, int]


class NoAvailableBranchError(RuntimeError):
    """Raised when no non-terminal branch is available."""

    def __init__(self) -> None:
        """Initialize the error for missing non-terminal branches."""
        super().__init__("No non-terminal branch is available.")


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


def make_branch_sequence_factory() -> list[BranchKey]:
    """Create a factory for generating branch sequences.

    Returns:
        list[BranchKey]: An empty list that can be used to store branch sequences.

    """
    return []


def make_branches_sorted_by_value_factory() -> dict[BranchKey, BranchSortValue]:
    """Create a factory for generating branches sorted by value.

    Returns:
        dict[BranchKey, BranchSortValue]: An empty dictionary that can be used to store branches sorted by their values.

    """
    return {}


def make_branch_frontier_factory() -> BranchFrontierState:
    """Create the generic frontier bookkeeping helper for one node."""
    return BranchFrontierState()


def make_default_objective() -> Objective[TurnState]:
    """Create the default objective preserving current adversarial semantics."""
    return AdversarialZeroSumObjective()


@dataclass(slots=True)
class NodeMinmaxEvaluation[
    NodeWithValueT: NodeWithValue = NodeWithValue,
    StateT: TurnState = TurnState,
]:
    r"""Value-first minimax evaluation attached to a tree node."""

    # a reference to the original tree node that is evaluated
    tree_node: TreeNode[NodeWithValueT, StateT]

    # canonical state-evaluator value
    direct_value: Value | None = None

    # canonical minimax value computed from descendants
    minmax_value: Value | None = None

    @property
    def backed_up_value(self) -> Value | None:
        """Return the generic backed-up Value alias used by game-agnostic APIs."""
        return self.minmax_value

    @backed_up_value.setter
    def backed_up_value(self, value: Value | None) -> None:
        """Set the generic backed-up Value alias used by game-agnostic APIs."""
        self.minmax_value = value

    # the sequence of best branches from this node
    best_branch_sequence: list[BranchKey] = field(
        default_factory=make_branch_sequence_factory
    )

    # version of principal variation content for change detection
    pv_version: int = 0

    # cached pv_version of the current best child at last PV materialization
    _pv_cached_best_child_version: int | None = None

    # the children of the tree node are kept in a dictionary that can be sorted by their evaluations ()

    # children_sorted_by_value records subjective values of children by descending order
    # subjective value means the values is from the point of view of player_to_branch
    # careful, I have hard coded in the self.best_child() function the descending order for
    # fast access to the best element, so please do not change!
    branches_sorted_by_value_: dict[BranchKey, BranchSortValue] = field(
        default_factory=make_branches_sorted_by_value_factory
    )

    # convention of descending order, careful if changing read above!!
    best_index_for_value: int = 0

    # Child branches that remain relevant to future search.
    branch_frontier: BranchFrontierState = field(
        default_factory=make_branch_frontier_factory
    )

    # policy used to orchestrate backup behavior from updated children
    backup_policy: "BackupPolicy[Any] | None" = None

    # ordering policy for comparing/projecting Value objects
    evaluation_ordering: EvaluationOrdering = DEFAULT_EVALUATION_ORDERING

    # objective responsible for semantic interpretation of Value objects at this node
    objective: Objective[StateT] = field(default_factory=make_default_objective)

    @property
    def branches_sorted_by_value(self) -> dict[BranchKey, BranchSortValue]:
        """Return a dictionary containing the branches of the node sorted by their values.

        Returns:
            dict[BranchKey, BranchSortValue]: A dictionary where the keys are the branches in the node and
            the values are the corresponding sort values.

        """
        return self.branches_sorted_by_value_

    @property
    def branches_to_explore(self) -> list[BranchKey]:
        """Backward-compatible ordered view of the current branch frontier."""
        return self.frontier_branches_in_order()

    def get_value(self) -> Value:
        """Return the canonical value for this node.

        Returns:
            Value: The backed-up value when available, otherwise direct value.

        """
        return canonical_value.get_value(
            backed_up_value=self.backed_up_value,
            direct_value=self.direct_value,
        )

    def get_score(self) -> float:
        """Return the canonical scalar score for this node.

        Consumer code should use this and ``get_value_candidate``/``get_value``.
        Use this for scalar access to canonical Value.
        """
        return canonical_value.get_score(
            backed_up_value=self.backed_up_value,
            direct_value=self.direct_value,
        )

    def get_value_candidate(self) -> Value | None:
        """Return backed-up value when available, else direct Value, or ``None``."""
        return canonical_value.get_value_candidate(
            backed_up_value=self.backed_up_value,
            direct_value=self.direct_value,
        )

    def has_exact_value(self) -> bool:
        """Return True when the candidate Value is exact."""
        return canonical_value.is_exact_value(self.get_value_candidate())

    def is_terminal(self) -> bool:
        """Return True when the candidate Value says this node's own state is terminal."""
        return canonical_value.is_terminal_value(self.get_value_candidate())

    def has_over_event(self) -> bool:
        """Return True when the candidate Value carries exact outcome metadata."""
        return canonical_value.has_over_event(self.get_value_candidate())

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

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the best available Value candidate for a child branch.

        Internal helper shared with backup policies during Step 7 migration.
        """
        child = self.tree_node.branches_children[branch_key]
        if child is None:
            return None

        child_eval = child.tree_evaluation
        return canonical_value.get_value_candidate(
            backed_up_value=child_eval.minmax_value,
            direct_value=child_eval.direct_value,
        )

    def best_branch(self) -> BranchKey | None:
        """Return the best branch node based on the subjective value.

        Returns:
            The best branch based on the subjective value, or None if there are no branch open.

        """
        ordered = self._decision_ordered_branches()
        if not ordered:
            return None
        return ordered[0]

    def best_branch_to_explore(self) -> BranchKey:
        """Return the best branch that remains unresolved.

        Returns:
            The best branch that remains worth exploring.

        Raises:
            Exception: If no unresolved branch is available.

        """
        branch_key: BranchKey
        for branch_key in self.branches_sorted_by_value:
            child = self.tree_node.branches_children[branch_key]
            assert child is not None
            if not child.tree_evaluation.has_exact_value():
                return branch_key
        raise NoAvailableBranchError

    def best_branch_value(self) -> BranchSortValue | None:
        """Return the value of the best branch.

        If the `branches_sorted_by_value` dictionary is not empty, it returns the value of the first child node with
        the highest subjective value. Otherwise, it returns None.

        Returns:
            BranchSortValue | None: The sort value of the best branch, or None if there are no opened branches.

        """
        best_value: BranchSortValue | None
        # fast way to access first key with the highest subjective value
        if self.branches_sorted_by_value:
            best_value = next(iter(self.branches_sorted_by_value.values()))
        else:
            best_value = None
        return best_value

    def second_best_branch(self) -> BranchKey:
        """Return the second-best branch based on the subjective value.

        Returns:
            The second-best branch.

        """
        ordered = self._decision_ordered_branches()
        assert len(ordered) >= 2
        return ordered[1]

    def _decision_ordered_branches(self) -> list[BranchKey]:
        """Return branches ordered for best-choice decisions.

        Uses semantic_compare on Value for correctness, with previous search-order
        tuple as deterministic tie-breaker.
        """
        candidates: list[tuple[BranchKey, Value, BranchSortValue]] = []
        for branch_key, sort_value in self.branches_sorted_by_value.items():
            child_value = self.child_value_candidate(branch_key)
            if child_value is None:
                continue
            candidates.append((branch_key, child_value, sort_value))

        def _cmp(
            a: tuple[BranchKey, Value, BranchSortValue],
            b: tuple[BranchKey, Value, BranchSortValue],
        ) -> int:
            sem = self.objective.semantic_compare(a[1], b[1], self.tree_node.state)
            if sem != 0:
                return -sem
            if a[2] < b[2]:
                return -1
            if a[2] > b[2]:
                return 1
            return -1 if str(a[0]) < str(b[0]) else (1 if str(a[0]) > str(b[0]) else 0)

        return [item[0] for item in sorted(candidates, key=cmp_to_key(_cmp))]

    def get_over_event_candidate(self) -> OverEvent | None:
        """Return exact outcome metadata from the candidate Value when present."""
        return canonical_value.get_over_event_candidate(self.get_value_candidate())

    def is_over(self) -> bool:
        """Return True only when this node's own state is terminal."""
        return self.is_terminal()

    def is_win(self) -> bool:
        """Return whether the exact candidate outcome is a win."""
        over_event = self.get_over_event_candidate()
        if over_event is not None:
            return over_event.is_win()
        return False

    def is_draw(self) -> bool:
        """Return whether the exact candidate outcome is a draw."""
        over_event = self.get_over_event_candidate()
        if over_event is not None:
            return over_event.is_draw()
        return False

    @property
    def over_event(self) -> OverEvent | None:
        """Return exact outcome metadata from the candidate Value when present."""
        return self.get_over_event_candidate()

    def is_winner(self, player: Color) -> bool:
        """Return whether the exact candidate outcome names ``player`` as winner.

        Args:
            player (Color): The color of the player to check.

        Returns:
            bool: True if the player is the winner, False otherwise.

        """
        over_event = self.get_over_event_candidate()
        if over_event is not None:
            return over_event.is_winner(player)
        return False

    def print_branches_sorted_by_value(
        self, dynamics: SearchDynamics[Any, Any]
    ) -> None:
        """Print the branches sorted by their subjective sort value.

        The method iterates over the branch_sorted_by_value dictionary and prints each branch along with its
        subjective sort value. The output is formatted as follows:
        "<branch>: <subjective_sort_value> $$"

        Returns:
            None

        """
        print(
            "here are the ",
            len(self.branches_sorted_by_value),
            " branch sorted by value: ",
        )
        branch_key: BranchKey
        for branch_key, subjective_sort_value in self.branches_sorted_by_value.items():
            print(
                dynamics.action_name(self.tree_node.state, branch_key),
                subjective_sort_value[0],
                end=" $$ ",
            )
        print("")

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

    def print_branches_to_explore(self) -> None:
        """Backward-compatible wrapper for printing frontier branches."""
        self.print_frontier_branches()

    def print_frontier_branches(self) -> None:
        """Print the unresolved branches that remain worth exploring.

        Returns:
            None

        """
        frontier_branches = self.frontier_branches_in_order()
        print(
            "here are the ",
            len(frontier_branches),
            " frontier branches: ",
            end=" ",
        )
        for branch in frontier_branches:
            print(branch, end=" ")
        print(" ")

    def print_info(self, dynamics: SearchDynamics[Any, Any]) -> None:
        """Print information about the node.

        This method prints the ID of the node, the branches of its children, the children sorted by value,
        and the unresolved children that remain worth exploring.
        """
        print("Soy el Node", self.tree_node.id)
        self.tree_node.print_branches_children()
        self.print_branches_sorted_by_value(dynamics=dynamics)
        self.print_frontier_branches()
        # TODO: probably more to print...

    def record_sort_value_of_child(self, branch_key: BranchKey) -> None:
        """Store the subjective value of the branch in self.branches_sorted_by_value (automatically sorted).

        Args:
            branch_key (BranchKey): The branch key whose value needs to be recorded.

        Returns:
            None

        """
        # - branches_sorted_by_value records subjective value of branches by descending order
        # therefore we have to convert the value_white of children into a subjective value that depends
        # on the player to branch
        # - subjective best branch/children is at index 0 however sortedValueDict are sorted ascending (best index: -1),
        # therefore for white we have negative values
        child = self.tree_node.branches_children[branch_key]
        assert child is not None
        child_value = self.child_value_candidate(branch_key)
        assert child_value is not None, (
            f"Cannot record sort value: child {branch_key} has no Value yet. "
            "Ensure children receive direct_value explicitly as a Value or "
            "minmax_value via backup before calling record_sort_value_of_child()."
        )
        self.branches_sorted_by_value_[branch_key] = self.branch_sort_value(branch_key)

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
        for branch_key in branches_to_consider:
            self.record_sort_value_of_child(branch_key=branch_key)
        self.branches_sorted_by_value_ = sort_dic(self.branches_sorted_by_value_)

    def on_branch_opened(self, branch: BranchKey) -> None:
        """Record that a child branch has entered the frontier."""
        self.branch_frontier.on_branch_opened(branch)

    def has_frontier_branches(self) -> bool:
        """Return whether some child branches remain search-relevant."""
        return self.branch_frontier.has_frontier_branches()

    def frontier_branches_in_order(self) -> list[BranchKey]:
        """Return frontier branches ordered by current child-preference semantics."""
        return self.branch_frontier.ordered_frontier_branches(
            (*self.branches_sorted_by_value, *self.tree_node.branches_children)
        )

    def _branch_is_frontier_relevant(self, branch_key: BranchKey) -> bool:
        """Return whether a child branch can still affect future search results."""
        child = self.tree_node.branches_children.get(branch_key)
        return child is not None and not child.tree_evaluation.has_exact_value()

    def sync_branch_frontier(self, branches_to_refresh: set[BranchKey]) -> None:
        """Refresh frontier bookkeeping from updated child values."""
        if self.has_exact_value():
            self.branch_frontier.clear()
            return

        self.branch_frontier.sync_with_current_state(
            branches_to_refresh=branches_to_refresh,
            should_remain_in_frontier=self._branch_is_frontier_relevant,
        )

    def sync_branches_to_explore(self, branches_to_refresh: set[BranchKey]) -> None:
        """Backward-compatible wrapper for frontier synchronization."""
        self.sync_branch_frontier(branches_to_refresh)

    def sort_branches_to_explore(self) -> list[BranchKey]:
        """Backward-compatible wrapper returning ordered frontier branches.

        Returns:
            A sorted list of branches that remain worth exploring.

        """
        return self.frontier_branches_in_order()

    def update_best_branch_sequence(
        self, branches_with_updated_best_branch_seq: set[BranchKey]
    ) -> bool:
        """Update the best branch sequence based on notifications from children nodes.

        This uses the branch keys that identify children which updated their best-branch sequence.
        It only propagates a tail update when the current PV head already matches
        ``best_branch()``; otherwise this method is a no-op.

        Args:
            branches_with_updated_best_branch_seq (set[Ibranch]): A set of branch that have
                notified an updated best-branch sequence.

        Returns:
            bool: True if self.best_branch_sequence is modified, False otherwise.

        """
        best_branch_key = self.best_branch()
        if best_branch_key is None:
            return self.clear_best_branch_sequence()

        if best_branch_key not in branches_with_updated_best_branch_seq:
            return False

        best_node = self.tree_node.branches_children.get(best_branch_key)
        if best_node is None:
            return False

        if self.best_branch_sequence:
            if self.best_branch_sequence[0] != best_branch_key:
                return False

            best_child_version = int(
                getattr(best_node.tree_evaluation, "pv_version", 0)
            )
            if self.pv_cached_best_child_version == best_child_version:
                return False

        return self.set_best_branch_sequence(
            [
                best_branch_key,
                *best_node.tree_evaluation.best_branch_sequence,
            ]
        )

    def _set_best_branch_sequence(self, new_seq: list[BranchKey]) -> bool:
        """Set PV content and update cached child version.

        Returns:
            bool: True iff the PV content changed.

        """
        if self.best_branch_sequence == new_seq:
            return False

        self.best_branch_sequence = new_seq.copy()
        self.pv_version += 1

        if not self.best_branch_sequence:
            self._pv_cached_best_child_version = None
            return True

        best_branch_key = self.best_branch_sequence[0]
        best_child = self.tree_node.branches_children.get(best_branch_key)
        if best_child is None:
            self._pv_cached_best_child_version = None
            return True

        self._pv_cached_best_child_version = int(
            getattr(best_child.tree_evaluation, "pv_version", 0)
        )
        return True

    def set_best_branch_sequence(self, new_seq: list[BranchKey]) -> bool:
        """Public PV setter for backup policies/tests. Bumps pv_version only on content change."""
        return self._set_best_branch_sequence(new_seq)

    def clear_best_branch_sequence(self) -> bool:
        """Public PV clearer for backup policies/tests."""
        return self._set_best_branch_sequence([])

    @property
    def pv_cached_best_child_version(self) -> int | None:
        """Cached pv_version of the best child when PV was last materialized."""
        return self._pv_cached_best_child_version

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
        has_best_branch_seq_changed = self._set_best_branch_sequence(
            [
                best_branch_key,
                *best_child.tree_evaluation.best_branch_sequence,
            ]
        )
        assert self.best_branch_sequence
        return has_best_branch_seq_changed

    def is_value_subjectively_better_than_evaluation(self, value: Value) -> bool:
        """Check if the given value_white is subjectively better than the value_white_evaluator.

        Args:
            value (Value): The value to compare with the direct evaluator value.

        Returns:
            bool: True if the value_white is subjectively better than the value_white_evaluator, False otherwise.

        """
        assert self.direct_value is not None
        return self.child_is_better_than_direct(
            value,
            self.direct_value,
            side_to_move=self.tree_node.state.turn,
        )

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
