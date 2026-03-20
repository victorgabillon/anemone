"""Provide a small single-agent max node evaluation implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from valanga import BranchKey, State
from valanga.evaluations import Certainty

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

    objective: Objective[StateT] | None = field(default_factory=make_default_objective)

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
            self.required_objective.evaluate_value(child_value, self.tree_node.state),
            exactness_tie_break,
            child.tree_node.id,
        )

    def _exact_line_tactical_quality(
        self,
        *,
        child_value: Any,
        pv_length: int,
    ) -> tuple[int, int]:
        """Return the single-agent tactical quality beyond the raw child Value.

        Exact values carry meaningful proof-line quality, so PV length refines
        their equality class. Estimate values do not, so their tactical quality
        stays neutral regardless of current PV length.
        """
        if child_value.certainty in {Certainty.TERMINAL, Certainty.FORCED}:
            return (0, pv_length)
        return (1, 0)

    def _branch_tactical_quality_key(self, branch: BranchKey) -> tuple[int, int]:
        """Return the single-agent tactical-quality key used for exact equality."""
        child_value = self.child_value_candidate(branch)
        assert child_value is not None
        child_tree_evaluation = self.child_tree_evaluation(branch)
        assert child_tree_evaluation is not None
        return self._exact_line_tactical_quality(
            child_value=child_value,
            pv_length=len(child_tree_evaluation.best_branch_sequence),
        )
