"""Provide a small single-agent max node evaluation implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from anemone._valanga_types import AnyTurnState
from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.node_evaluation.tree.node_tree_evaluation import (
    BackupPolicyFactory,
    NodeTreeEvaluationState,
)
from anemone.objectives.single_agent_max import SingleAgentMaxObjective

if TYPE_CHECKING:
    from valanga import BranchKey

    from anemone.node_evaluation.tree.decision_ordering import BranchOrderingKey
    from anemone.objectives import Objective


def make_default_objective() -> Objective[AnyTurnState]:
    """Create the default single-agent objective."""
    return SingleAgentMaxObjective()


def make_default_backup_policy() -> ExplicitMaxBackupPolicy:
    """Create the default single-agent max backup policy."""
    return ExplicitMaxBackupPolicy()


@dataclass(slots=True)
class NodeMaxEvaluation[StateT: AnyTurnState = AnyTurnState](
    NodeTreeEvaluationState[Any, StateT]
):
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
        return (
            self.required_objective.evaluate_value(child_value, self.tree_node.state),
            self._branch_exact_line_tactical_quality(branch_key),
            child.tree_node.id,
        )
