"""Thin single-agent wrapper over the shared tree-evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol, Self

from anemone._valanga_types import AnyTurnState
from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.node_evaluation.tree.node_tree_evaluation import (
    BackupPolicyFactory,
    NodeTreeEvaluationState,
    TreeEvaluationChild,
)
from anemone.objectives.single_agent_max import SingleAgentMaxObjective

if TYPE_CHECKING:
    from anemone.nodes.tree_node import TreeNode
    from anemone.objectives import Objective


class NodeWithMaxEvaluation(TreeEvaluationChild[AnyTurnState], Protocol):
    """Recursive child-node protocol for the single-agent max wrapper."""

    @property
    def tree_evaluation(self) -> NodeMaxEvaluation:
        """Return the single-agent max evaluation associated with this node."""
        ...

    tree_node: TreeNode[Self, AnyTurnState]


def make_default_objective() -> Objective[AnyTurnState]:
    """Create the default single-agent objective."""
    return SingleAgentMaxObjective()


def make_default_backup_policy() -> ExplicitMaxBackupPolicy:
    """Create the default single-agent max backup policy."""
    return ExplicitMaxBackupPolicy()


@dataclass(slots=True)
class NodeMaxEvaluation[
    StateT: AnyTurnState = AnyTurnState,
    NodeWithMaxEvaluationT: NodeWithMaxEvaluation = NodeWithMaxEvaluation,
](NodeTreeEvaluationState[NodeWithMaxEvaluationT, StateT]):
    """Semantic max-search preset layered on ``NodeTreeEvaluationState``.

    The shared base class owns the generic tree-search machinery. This wrapper
    keeps the single-agent family name plus its default objective and backup
    policy so callers do not need to assemble the generic pieces manually.
    """

    _default_backup_policy_factory: ClassVar[BackupPolicyFactory | None] = (
        make_default_backup_policy
    )

    # Family wrapper installs single-agent max semantics on the shared engine.
    objective: Objective[StateT] | None = field(default_factory=make_default_objective)
