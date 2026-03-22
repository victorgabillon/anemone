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
