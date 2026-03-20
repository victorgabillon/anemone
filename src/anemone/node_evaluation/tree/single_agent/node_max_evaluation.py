"""Provide a small single-agent max node evaluation implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from valanga import BranchKey, State
from valanga.evaluations import Certainty, Value

from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.node_evaluation.tree.node_tree_evaluation import (
    BackupPolicyFactory,
    NodeTreeEvaluationState,
)
from anemone.objectives.single_agent_max import SingleAgentMaxObjective

if TYPE_CHECKING:
    from anemone.node_evaluation.tree.decision_ordering import BranchOrderingKey
    from anemone.objectives import Objective


def make_default_objective() -> Objective[State]:
    """Create the default single-agent objective."""
    return SingleAgentMaxObjective()


def make_default_backup_policy() -> ExplicitMaxBackupPolicy:
    """Create the default single-agent max backup policy."""
    return ExplicitMaxBackupPolicy()


@dataclass(slots=True)
class NodeMaxEvaluation[StateT: State = State](NodeTreeEvaluationState[Any, StateT]):
    """Canonical Value-based node evaluation for single-agent max search."""

    _default_backup_policy_factory: ClassVar[BackupPolicyFactory | None] = (
        make_default_backup_policy
    )

    objective: Objective[StateT] = field(default_factory=make_default_objective)

    def branch_sort_value(self, branch_key: BranchKey) -> BranchOrderingKey:
        """Return the shared branch-ordering key for one single-agent child."""
        child = self.tree_node.branches_children[branch_key]
        assert child is not None
        child_value = self.child_value_candidate(branch_key)
        assert child_value is not None
        exactness_tie_break = (
            0 if child_value.certainty in {Certainty.TERMINAL, Certainty.FORCED} else 1
        )
        return (
            self.objective.evaluate_value(child_value, self.tree_node.state),
            exactness_tie_break,
            child.tree_node.id,
        )

    def _decision_semantic_compare(self, left: Value, right: Value) -> int:
        """Compare child values using current single-agent decision semantics."""
        return self.objective.semantic_compare(left, right, self.tree_node.state)

    def _branch_values_are_equal(
        self,
        *,
        branch: BranchKey,
        best_branch: BranchKey,
    ) -> bool:
        """Return whether two child Values are exactly equal for max semantics."""
        branch_value = self.child_value_candidate(branch)
        best_value = self.child_value_candidate(best_branch)
        assert branch_value is not None
        assert best_value is not None
        return branch_value == best_value

    def _branch_values_are_considered_equal(
        self,
        *,
        branch: BranchKey,
        best_branch: BranchKey,
    ) -> bool:
        """Return whether two child Values are tied under max decision semantics."""
        branch_value = self.child_value_candidate(branch)
        best_value = self.child_value_candidate(best_branch)
        assert branch_value is not None
        assert best_value is not None
        return (
            self.objective.semantic_compare(
                branch_value,
                best_value,
                self.tree_node.state,
            )
            == 0
        )

    def update_branches_values(self, branches_to_consider: set[BranchKey]) -> None:
        """Incrementally refresh ordering keys for the currently-valued children."""
        self.decision_ordering.update_branch_ordering_keys(
            {
                branch_key
                for branch_key in branches_to_consider
                if self.child_value_candidate(branch_key) is not None
            },
            branch_ordering_key_getter=self.branch_sort_value,
        )

    def _ensure_decision_ordering_ready(self) -> None:
        """Max now relies on explicit incremental ordering maintenance."""
