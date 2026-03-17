"""Provide a small single-agent max node evaluation implementation."""

from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Any, cast

from valanga import BranchKey, OverEvent, State
from valanga.evaluations import Value

from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.backup_policies.protocols import BackupPolicy
from anemone.node_evaluation.common import canonical_value
from anemone.node_evaluation.common.branch_frontier import BranchFrontierState
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import Objective
from anemone.objectives.single_agent_max import SingleAgentMaxObjective


def make_branch_sequence_factory() -> list[BranchKey]:
    """Create a factory for best-branch sequences."""
    return []


def make_default_objective() -> Objective[State]:
    """Create the default single-agent objective."""
    return SingleAgentMaxObjective()


def make_default_backup_policy() -> ExplicitMaxBackupPolicy:
    """Create the default single-agent max backup policy."""
    return ExplicitMaxBackupPolicy()


def make_branch_frontier_factory() -> BranchFrontierState:
    """Create the generic frontier bookkeeping helper for one node."""
    return BranchFrontierState()


@dataclass(slots=True)
class NodeMaxEvaluation[StateT: State = State]:
    """Canonical Value-based node evaluation for single-agent max search."""

    tree_node: TreeNode[Any, StateT]
    direct_value: Value | None = None
    _backed_up_value: Value | None = None
    best_branch_sequence: list[BranchKey] = field(
        default_factory=make_branch_sequence_factory
    )
    objective: Objective[StateT] = field(default_factory=make_default_objective)
    backup_policy: BackupPolicy["NodeMaxEvaluation[StateT]"] = field(
        default_factory=make_default_backup_policy
    )
    branch_frontier: BranchFrontierState = field(
        default_factory=make_branch_frontier_factory
    )

    @property
    def backed_up_value(self) -> Value | None:
        """Return the canonical backed-up value for this node."""
        return self._backed_up_value

    @backed_up_value.setter
    def backed_up_value(self, value: Value | None) -> None:
        """Set the canonical backed-up value for this node."""
        self._backed_up_value = value

    def get_value_candidate(self) -> Value | None:
        """Return backed-up value when available, else direct value."""
        return canonical_value.get_value_candidate(
            backed_up_value=self.backed_up_value,
            direct_value=self.direct_value,
        )

    def get_value(self) -> Value:
        """Return the canonical Value for this node."""
        return canonical_value.get_value(
            backed_up_value=self.backed_up_value,
            direct_value=self.direct_value,
        )

    def get_score(self) -> float:
        """Return the canonical scalar score for this node."""
        return canonical_value.get_score(
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

    @property
    def over_event(self) -> OverEvent | None:
        """Return exact outcome metadata when present on the canonical candidate."""
        return canonical_value.get_over_event_candidate(self.get_value_candidate())

    def child_value_candidate(self, branch_key: BranchKey) -> Value | None:
        """Return the best available Value candidate for a child branch."""
        child = self.tree_node.branches_children[branch_key]
        if child is None:
            return None
        return cast("Value | None", child.tree_evaluation.get_value_candidate())

    def _decision_ordered_branches(self) -> list[BranchKey]:
        """Return child branches ordered by the current single-agent preference."""
        candidates: list[tuple[BranchKey, Value, int]] = []
        for branch_key, child in self.tree_node.branches_children.items():
            if child is None:
                continue
            child_value = child.tree_evaluation.get_value_candidate()
            if child_value is None:
                continue
            candidates.append((branch_key, child_value, child.tree_node.id))

        if not candidates:
            return []

        def _cmp(
            left: tuple[BranchKey, Value, int],
            right: tuple[BranchKey, Value, int],
        ) -> int:
            semantic = self.objective.semantic_compare(
                left[1],
                right[1],
                self.tree_node.state,
            )
            if semantic > 0:
                return -1
            if semantic < 0:
                return 1
            if left[2] < right[2]:
                return -1
            if left[2] > right[2]:
                return 1
            return (
                -1
                if str(left[0]) < str(right[0])
                else (1 if str(left[0]) > str(right[0]) else 0)
            )

        return [
            branch_key for branch_key, _, _ in sorted(candidates, key=cmp_to_key(_cmp))
        ]

    def best_branch(self) -> BranchKey | None:
        """Return the best currently-valued child branch."""
        ordered = self._decision_ordered_branches()
        if not ordered:
            return None
        return ordered[0]

    def on_branch_opened(self, branch: BranchKey) -> None:
        """Record that a child branch has entered the frontier."""
        self.branch_frontier.on_branch_opened(branch)

    def has_frontier_branches(self) -> bool:
        """Return whether some child branches remain search-relevant."""
        return self.branch_frontier.has_frontier_branches()

    def frontier_branches_in_order(self) -> list[BranchKey]:
        """Return frontier branches ordered by current child-preference semantics."""
        return self.branch_frontier.ordered_frontier_branches(
            (*self._decision_ordered_branches(), *self.tree_node.branches_children)
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

    def update_best_branch_sequence(
        self, branches_with_updated_best_branch_seq: set[BranchKey]
    ) -> bool:
        """Update the current principal variation from the best child when needed."""
        best_branch_key = self.best_branch()
        if best_branch_key is None:
            return self.set_best_branch_sequence([])

        if (
            best_branch_key not in branches_with_updated_best_branch_seq
            and self.best_branch_sequence
            and self.best_branch_sequence[0] == best_branch_key
        ):
            return False

        best_child = self.tree_node.branches_children[best_branch_key]
        assert best_child is not None
        return self.set_best_branch_sequence(
            [best_branch_key, *best_child.tree_evaluation.best_branch_sequence]
        )

    def set_best_branch_sequence(self, new_seq: list[BranchKey]) -> bool:
        """Replace the stored principal variation content."""
        if self.best_branch_sequence == new_seq:
            return False
        self.best_branch_sequence = new_seq.copy()
        return True

    def backup_from_children(
        self,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> Any:
        """Delegate backup work to the configured single-agent backup policy."""
        return self.backup_policy.backup_from_children(
            node_eval=self,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )
