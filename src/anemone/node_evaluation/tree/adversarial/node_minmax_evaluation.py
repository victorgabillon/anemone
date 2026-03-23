"""Thin adversarial wrapper over the shared tree-evaluation engine."""
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol, Self

from valanga.evaluations import Value

from anemone._valanga_types import AnyTurnState
from anemone.node_evaluation.tree.node_tree_evaluation import (
    BackupPolicyFactory,
    NodeTreeEvaluationState,
    TreeEvaluationChild,
)
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import AdversarialZeroSumObjective, Objective

if TYPE_CHECKING:
    from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy


class NodeWithValue(TreeEvaluationChild[AnyTurnState], Protocol):
    """Recursive child-node protocol for the minimax wrapper.

    ``TreeEvaluationChild`` captures the generic engine's needs, but the named
    minimax wrapper still benefits from expressing that its ``tree_node`` points
    at more nodes from the same semantic family.
    """

    @property
    def tree_evaluation(self) -> "NodeMinmaxEvaluation":
        """Return the minimax evaluation associated with this node."""
        ...

    tree_node: TreeNode[Self, AnyTurnState]


def make_default_objective() -> Objective[AnyTurnState]:
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
    StateT: AnyTurnState = AnyTurnState,
](NodeTreeEvaluationState[NodeWithValueT, StateT]):
    """Semantic minimax preset layered on ``NodeTreeEvaluationState``.

    The shared base class owns the generic tree-search mechanics. This wrapper
    keeps the adversarial family name, default objective, and the meaningful
    ``minmax_value`` vocabulary expected by callers.
    """

    _default_backup_policy_factory: ClassVar[BackupPolicyFactory | None] = (
        make_default_backup_policy
    )

    # Family wrapper installs adversarial value semantics on the shared engine.
    objective: Objective[StateT] | None = field(default_factory=make_default_objective)

    @property
    def minmax_value(self) -> Value | None:
        """Return the adversarial family's semantic alias for ``backed_up_value``."""
        return self.backed_up_value

    @minmax_value.setter
    def minmax_value(self, value: Value | None) -> None:
        """Set the adversarial family's semantic alias for ``backed_up_value``."""
        self.backed_up_value = value
