"""Provide Value-first NodeMinmaxEvaluation implementation for minimax search."""

# TODO: maybe further split values from over?

from dataclasses import dataclass, field
from functools import cmp_to_key
from math import log
from random import choice
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

from valanga import (
    BranchKey,
    Color,
    FloatyStateEvaluation,
    ForcedOutcome,
    OverEvent,
    StateEvaluation,
    TurnState,
)

from anemone.dynamics import SearchDynamics
from anemone.nodes.itree_node import ITreeNode
from anemone.nodes.tree_node import TreeNode
from anemone.utils.logger import anemone_logger
from anemone.utils.my_value_sorted_dict import sort_dic
from anemone.values import (
    DEFAULT_EVALUATION_ORDERING,
    Certainty,
    EvaluationOrdering,
    Value,
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


def make_branches_not_over_factory() -> list[BranchKey]:
    """Create a factory for generating branches that are not over.

    Returns:
        list[BranchKey]: An empty list that can be used to store branches that are not over.

    """
    return []


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

    # the list of branches that have not yet be found to be over
    # using atm a list instead of set as atm python set are not insertion ordered which adds randomness
    # and makes debug harder
    branches_not_over: list[BranchKey] = field(
        default_factory=make_branches_not_over_factory
    )

    # creating a base Over event that is set to None
    over_event: OverEvent = field(default_factory=OverEvent)

    # policy used to orchestrate backup behavior from updated children
    backup_policy: "BackupPolicy | None" = None

    # ordering policy for comparing/projecting Value objects
    evaluation_ordering: EvaluationOrdering = DEFAULT_EVALUATION_ORDERING

    @property
    def branches_sorted_by_value(self) -> dict[BranchKey, BranchSortValue]:
        """Return a dictionary containing the branches of the node sorted by their values.

        Returns:
            dict[BranchKey, BranchSortValue]: A dictionary where the keys are the branches in the node and
            the values are the corresponding sort values.

        """
        return self.branches_sorted_by_value_

    def get_value(self) -> Value:
        """Return the canonical value for this node.

        Returns:
            Value: The minmax value when available, otherwise direct value.

        """
        if self.minmax_value is not None:
            return self.minmax_value
        assert self.direct_value is not None, (
            "Node has no canonical Value: both minmax_value and direct_value are None. "
            "Set direct_value via set_evaluation() or compute minmax_value via backup."
        )
        return self.direct_value

    def get_score(self) -> float:
        """Return the canonical scalar score for this node.

        Consumer code should use this and ``get_value_candidate``/``get_value``.
        Use this for scalar access to canonical Value.
        """
        return self.get_value().score

    def get_value_candidate(self) -> Value | None:
        """Return minmax when available, else direct Value, or ``None``."""
        if self.minmax_value is not None:
            return self.minmax_value
        return self.direct_value

    def require_value_candidate(self) -> Value:
        """Return best candidate Value, raising when no Value exists yet."""
        value = self.get_value_candidate()
        assert value is not None, (
            "Node has no candidate Value: both minmax_value and direct_value are None."
        )
        return value

    def is_terminal_candidate(self) -> bool:
        """Return True when candidate Value is TERMINAL/FORCED and has ``over_event``."""
        value = self.get_value_candidate()
        return (
            value is not None
            and value.certainty in (Certainty.TERMINAL, Certainty.FORCED)
            and value.over_event is not None
        )

    def sync_over_from_values(self) -> None:
        """Keep ``over_event`` aligned with terminal/forced Value metadata.

        Non-terminal values do not clear existing over state; this keeps legacy
        ``update_over`` behavior stable when Value metadata is not terminal.
        """
        value = self.get_value_candidate()
        if (
            value is not None
            and value.certainty in {Certainty.TERMINAL, Certainty.FORCED}
            and value.over_event is not None
            and hasattr(value.over_event, "is_over")
        ):
            self.over_event = value.over_event

    def set_evaluation(self, evaluation: float) -> None:
        """Set the evaluation from the state evaluator.

        Args:
            evaluation (float): The evaluation value to be set.

        Returns:
            None

        """
        # legacy API: convert float into Value representation.
        if (
            self.direct_value is None
            or self.direct_value.certainty is not Certainty.TERMINAL
        ):
            self.direct_value = Value(
                score=evaluation,
                certainty=Certainty.ESTIMATE,
                over_event=None,
            )

        # Keep leaf minimax aligned with the latest direct evaluation.
        if not self.tree_node.branches_children or self.minmax_value is None:
            self.minmax_value = self.direct_value
        self.sync_over_from_values()

    def child_is_better_than_direct(
        self, child: Value, direct: Value, *, side_to_move: Color
    ) -> bool:
        """Determine if a child's value is better than the direct evaluation for the current node."""
        return (
            self.evaluation_ordering.semantic_compare(
                child,
                direct,
                side_to_move=side_to_move,
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
        if child_eval.minmax_value is not None:
            return child_eval.minmax_value

        if child_eval.direct_value is not None:
            return child_eval.direct_value
        return None

    def _child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Backward-compat shim for older tests/callers.

        TODO: remove in Step 9 once all callers use ``child_value_candidate``.
        """
        return self.child_value_candidate(branch_key)

    def subjective_value_(self, value_white: float) -> float:
        """Return the subjective value of `value_white` from the point of view of the `self.tree_node.player_to_branch`.

        The subjective value is calculated based on the player to branch. If the player to branch is `Color.WHITE`, then the
        `value_white` is returned as is. Otherwise, the negative of `value_white` is returned.

        Args:
            value_white (float): The value from the point of view of the white player.

        Returns:
            float: The subjective value of `value_white` based on the player to branch.

        """
        return value_white if self.tree_node.state.turn is Color.WHITE else -value_white

    def subjective_value(self) -> float:
        """Return the subjective value of self.value_white from the point of view of the self.tree_node.player_to_branch.

        If the player to branch is Color.WHITE, the subjective value is the canonical score.
        If the player to branch is not Color.WHITE, the subjective value is the negated canonical score.

        Returns:
            float: The subjective value of self.value_white.

        """
        score = self.get_score()
        return score if self.tree_node.state.turn is Color.WHITE else -score

    def best_branch(self) -> BranchKey | None:
        """Return the best branch node based on the subjective value.

        Returns:
            The best branch based on the subjective value, or None if there are no branch open.

        """
        ordered = self._decision_ordered_branches()
        if not ordered:
            return None
        return ordered[0]

    def best_branch_not_over(self) -> BranchKey:
        """Return the best branch that is not leading to a game-over.

        Returns:
            The best branch that is not leading to a game-over.

        Raises:
            Exception: If no branch is found that is not over.

        """
        branch_key: BranchKey
        for branch_key in self.branches_sorted_by_value:
            child = self.tree_node.branches_children[branch_key]
            assert child is not None
            if not child.tree_evaluation.is_terminal_candidate():
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
            sem = self.evaluation_ordering.semantic_compare(
                a[1], b[1], side_to_move=self.tree_node.state.turn
            )
            if sem != 0:
                return -sem
            if a[2] < b[2]:
                return -1
            if a[2] > b[2]:
                return 1
            return -1 if str(a[0]) < str(b[0]) else (1 if str(a[0]) > str(b[0]) else 0)

        return [item[0] for item in sorted(candidates, key=cmp_to_key(_cmp))]

    def is_over(self) -> bool:
        """Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.

        """
        over_event = self.over_event
        if hasattr(over_event, "is_over"):
            return bool(over_event.is_over())
        return bool(getattr(over_event, "_is_over", False))

    def is_win(self) -> bool:
        """Check if the current game state is a win.

        Returns:
            bool: True if the game state is a win, False otherwise.

        """
        return self.over_event.is_win()

    def is_draw(self) -> bool:
        """Check if the current game state is a draw.

        Returns:
            bool: True if the game state is a draw, False otherwise.

        """
        return self.over_event.is_draw()

    def is_winner(self, player: Color) -> bool:
        """Determine if the specified player is the winner.

        Args:
            player (Color): The color of the player to check.

        Returns:
            bool: True if the player is the winner, False otherwise.

        """
        return self.over_event.is_winner(player)

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

    def print_branches_not_over(self) -> None:
        """Print the branches that are not over.

        This method prints the branches that are not marked as 'over'.
        It iterates over the `branches_not_over` list and prints each child's ID.

        Returns:
            None

        """
        print(
            "here are the ", len(self.branches_not_over), " branch not over: ", end=" "
        )
        for branch in self.branches_not_over:
            print(branch, end=" ")
        print(" ")

    def print_info(self, dynamics: SearchDynamics[Any, Any]) -> None:
        """Print information about the node.

        This method prints the ID of the node, the branches of its children, the children sorted by value,
        and the children that are not over.
        """
        print("Soy el Node", self.tree_node.id)
        self.tree_node.print_branches_children()
        self.print_branches_sorted_by_value(dynamics=dynamics)
        self.print_branches_not_over()
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
            "Ensure children receive direct_value via set_evaluation() or "
            "minmax_value via backup before calling record_sort_value_of_child()."
        )
        subjective_value_of_child = self.evaluation_ordering.search_sort_key(
            child_value,
            side_to_move=self.tree_node.state.turn,
        )
        if self.is_terminal_candidate():
            # the shorter the check the better now
            self.branches_sorted_by_value_[branch_key] = (
                subjective_value_of_child,
                -len(child.tree_evaluation.best_branch_sequence),
                child.tree_node.id,
            )

        else:
            # the longer the line the better now
            self.branches_sorted_by_value_[branch_key] = (
                subjective_value_of_child,
                len(child.tree_evaluation.best_branch_sequence),
                child.tree_node.id,
            )

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

    def becoming_over_from_children(self) -> None:
        """Switch the node to over status.

        This method is called when the node is requested to switch to the "over" status. It performs the necessary
        operations to update the node's status and determine the winner.

        Raises:
            AssertionError: If the node is already in the "over" status.

        """
        assert not self.is_terminal_candidate()

        # becoming over triggers a full update record_sort_value_of_child
        # where ties are now broken to reach over as fast as possible
        # TODO: we should reach it asap if we are winning and think about what to ddo in other scenarios....
        branch_key: BranchKey
        for branch_key in self.tree_node.branches_children:
            self.record_sort_value_of_child(branch_key=branch_key)

        best_over_branch_key = self._pick_over_child_branch()
        best_child = self.tree_node.branches_children[best_over_branch_key]
        assert best_child is not None
        best_child_value = best_child.tree_evaluation.require_value_candidate()
        assert best_child.tree_evaluation.is_terminal_candidate()
        assert best_child_value.over_event is not None
        child_over_event = best_child_value.over_event

        self.over_event.becomes_over(
            how_over=child_over_event.how_over,
            who_is_winner=child_over_event.who_is_winner,
            termination=child_over_event.termination,
        )

        self.minmax_value = Value(
            score=best_child_value.score,
            certainty=(
                best_child_value.certainty
                if best_child_value.certainty in (Certainty.TERMINAL, Certainty.FORCED)
                else Certainty.TERMINAL
            ),
            over_event=self.over_event,
        )
        self.sync_over_from_values()

    def _pick_over_child_branch(self) -> BranchKey:
        """Pick the terminal child branch that determines this node over event.

        Preference order is: winning over child for the current player, then draw,
        then any remaining terminal child. If search-order metadata is unavailable,
        fallback order is deterministic by branch key.

        Returns:
            BranchKey: The selected branch key.

        Raises:
            AssertionError: If no child is terminal while this method is called.

        """
        over_branches: list[BranchKey] = [
            branch_key
            for branch_key in self.branches_sorted_by_value
            if (
                (child := self.tree_node.branches_children[branch_key]) is not None
                and child.tree_evaluation.is_terminal_candidate()
            )
        ]
        if not over_branches:
            over_branches = sorted(
                [
                    branch_key
                    for branch_key, child in self.tree_node.branches_children.items()
                    if child is not None
                    and child.tree_evaluation.is_terminal_candidate()
                ],
                key=str,
            )

        assert over_branches, "becoming_over_from_children called with no over child"

        branch_key: BranchKey
        for branch_key in over_branches:
            child = self.tree_node.branches_children[branch_key]
            assert child is not None
            if child.tree_evaluation.is_winner(self.tree_node.state.turn):
                return branch_key

        for branch_key in over_branches:
            child = self.tree_node.branches_children[branch_key]
            assert child is not None
            if child.tree_evaluation.is_draw():
                return branch_key

        return over_branches[0]

    def update_over(self, branches_with_updated_over: set[BranchKey]) -> bool:
        """Update the over_event of the node based on notification of change of over_event in children.

        Args:
            branches_with_updated_over (set[BranchKey]): A set of branch keys linking to the children
                nodes that have been updated with their over_event.

        Returns:
            bool: True if the node has become newly over, False otherwise.

        """
        is_newly_over = False

        # Two cases can make this node (self) become over:
        # 1. One of the children of this node is over and is a win for the node.player_to_branch.
        # 2. All children are now over, then choose your best over event (choose draw if you can avoid a loss).

        for branch in branches_with_updated_over:
            child = self.tree_node.branches_children[branch]
            assert child is not None
            assert child.tree_evaluation.is_terminal_candidate()
            if branch in self.branches_not_over:
                self.branches_not_over.remove(branch)

            # Check if child is already not in children_not_over.
            if not self.is_terminal_candidate() and child.tree_evaluation.is_winner(
                self.tree_node.state.turn
            ):
                self.becoming_over_from_children()
                is_newly_over = True

        # Check if all children are over but not winning for self.tree_node.player_to_branch.
        has_non_over_child = any(
            child is not None and not child.tree_evaluation.is_terminal_candidate()
            for child in self.tree_node.branches_children.values()
        )
        if not has_non_over_child:
            self.branches_not_over.clear()

        if not self.is_terminal_candidate() and not has_non_over_child:
            self.becoming_over_from_children()
            is_newly_over = True

        if is_newly_over:
            assert self.is_terminal_candidate()

        return is_newly_over

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

    def sort_branches_not_over(self) -> list[BranchKey]:
        """Sort the branches that are not over based on their value.

        Returns:
            A sorted list of branches that are not over.

        """
        # TODO: looks like the determinism of the sort induces some determinisin the play like always
        #  playing the same actions when a lot of them have equal value: introduce some randomness?
        return [
            branch
            for branch in self.branches_sorted_by_value
            if branch in self.branches_not_over
        ]  # TODO: is this a fast way to do it?

    def update_value_minmax(self) -> None:
        """Legacy minimax update helper.

        Policy-driven production code should use ``backup_from_children``.
        Updates the minmax value for the current node based on the best child.
        """
        best_branch_key: BranchKey | None = self.best_branch()
        assert best_branch_key is not None
        best_child = self.tree_node.branches_children[best_branch_key]
        assert best_child is not None
        best_child_value = self.child_value_candidate(best_branch_key)
        assert best_child_value is not None, (
            "Cannot update minmax_value: best child has no Value candidate."
        )
        if self.tree_node.all_branches_generated:
            self.minmax_value = best_child_value
        else:
            if self.direct_value is None:
                self.minmax_value = best_child_value
                self.sync_over_from_values()
                return
            if self.child_is_better_than_direct(
                best_child_value,
                self.direct_value,
                side_to_move=self.tree_node.state.turn,
            ):
                self.minmax_value = best_child_value
            else:
                self.minmax_value = self.direct_value

        assert self.minmax_value is not None, (
            "minmax_value must be resolved from Value candidates during backup."
        )
        self.sync_over_from_values()

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
        policy: BackupPolicy = self.backup_policy
        return policy.backup_from_children(
            node_eval=self,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )

    def dot_description(self) -> str:
        """Return a string representation of the node's description in DOT format.

        The description includes canonical minmax/direct scores,
        as well as the best branch sequence and the over event tag.

        Returns:
            A string representation of the node's description in DOT format.

        """
        value_mm = (
            f"{self.minmax_value.score:.3f}"
            if self.minmax_value is not None
            else "None"
        )
        value_eval = (
            f"{self.direct_value.score:.3f}"
            if self.direct_value is not None
            else "None"
        )
        return (
            "\n wh_val_mm: "
            + value_mm
            + "\n wh_val_eval: "
            + value_eval
            + "\n branches*"
            + self.description_best_branch_sequence()
            + "\nover: "
            + self.over_event.get_over_tag()
        )

    def description_best_branch_sequence(self) -> str:
        """Return a string representation of the best branch sequence.

        This method iterates over the best node sequence and constructs a string representation
        of the branches in the sequence. Each branch is appended to the result string, separated by an underscore.

        Returns:
            A string representation of the best branch sequence.

        """
        res = ""
        branch_key: BranchKey
        for branch_key in self.best_branch_sequence:
            res += "_" + str(branch_key)
        return res

    def description_tree_visualizer_branch(self, child: ITreeNode[StateT]) -> str:
        """Return a string representation of the branch for the tree visualizer.

        Args:
            child (ITreeNode[StateT]): The child node representing the branch.

        Returns:
            str: A string representation of the branch for the tree visualizer.

        """
        _ = child  # to avoid unused variable warning, to be used when we want to print more info about the branch
        return ""

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

    def evaluate(self) -> StateEvaluation:
        """Build a StateEvaluation from current minmax state."""
        if self.is_terminal_candidate():
            value = self.require_value_candidate()
            assert value.over_event is not None
            return ForcedOutcome(
                outcome=value.over_event,
                line=self.best_branch_sequence.copy(),
            )
        return FloatyStateEvaluation(value_white=self.get_score())
