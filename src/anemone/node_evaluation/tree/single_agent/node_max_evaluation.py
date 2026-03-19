"""Provide a small single-agent max node evaluation implementation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from valanga import BranchKey, State

from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.backup_policies.protocols import BackupPolicy
from anemone.backup_policies.types import BackupResult
from anemone.node_evaluation.common.branch_ordering import (
    ordered_branches_from_candidates,
)
from anemone.node_evaluation.tree.node_tree_evaluation import NodeTreeEvaluationState
from anemone.objectives import Objective
from anemone.objectives.single_agent_max import SingleAgentMaxObjective

if TYPE_CHECKING:
    from valanga.evaluations import Value


def make_default_objective() -> Objective[State]:
    """Create the default single-agent objective."""
    return SingleAgentMaxObjective()


def make_default_backup_policy() -> ExplicitMaxBackupPolicy:
    """Create the default single-agent max backup policy."""
    return ExplicitMaxBackupPolicy()


@dataclass(slots=True)
class NodeMaxEvaluation[StateT: State = State](NodeTreeEvaluationState[Any, StateT]):
    """Canonical Value-based node evaluation for single-agent max search."""

    objective: Objective[StateT] = field(default_factory=make_default_objective)
    backup_policy: BackupPolicy["NodeMaxEvaluation[StateT]"] = field(
        default_factory=make_default_backup_policy
    )

    def decision_ordered_branches(self) -> list[BranchKey]:
        """Return child branches ordered by the current single-agent preference."""
        candidates: list[tuple[BranchKey, Value, int]] = []
        for branch_key, child in self.tree_node.branches_children.items():
            if child is None:
                continue
            child_value = child.tree_evaluation.get_value_candidate()
            if child_value is None:
                continue
            candidates.append((branch_key, child_value, child.tree_node.id))

        return ordered_branches_from_candidates(
            candidates,
            semantic_compare=lambda left, right: self.objective.semantic_compare(
                left,
                right,
                self.tree_node.state,
            ),
        )

    def best_branch(self) -> BranchKey | None:
        """Return the best currently-valued child branch."""
        ordered = self.decision_ordered_branches()
        if not ordered:
            return None
        return ordered[0]

    def _ordered_candidate_branches_for_frontier(self) -> tuple[BranchKey, ...]:
        """Return frontier candidates in current single-agent search order."""
        return (*self.decision_ordered_branches(), *self.tree_node.branches_children)

    def backup_from_children(
        self,
        branches_with_updated_value: set[BranchKey],
        branches_with_updated_best_branch_seq: set[BranchKey],
    ) -> BackupResult:
        """Delegate backup work to the configured single-agent backup policy."""
        return self.backup_policy.backup_from_children(
            node_eval=self,
            branches_with_updated_value=branches_with_updated_value,
            branches_with_updated_best_branch_seq=branches_with_updated_best_branch_seq,
        )
