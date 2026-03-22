"""Provide Value-first NodeMinmaxEvaluation implementation for minimax search."""
# pylint: disable=duplicate-code

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, runtime_checkable

from valanga import BranchKey
from valanga.evaluations import Value

from anemone._valanga_types import AnyTurnState
from anemone.node_evaluation.tree.decision_ordering import (
    BranchOrderingKey,
)
from anemone.node_evaluation.tree.node_tree_evaluation import (
    BackupPolicyFactory,
    NodeTreeEvaluationState,
    TreeEvaluationChild,
)
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import AdversarialZeroSumObjective, Objective

if TYPE_CHECKING:
    from anemone.backup_policies.explicit_minimax import ExplicitMinimaxBackupPolicy


@runtime_checkable
# Class created to avoid circular import and defines what is seen and needed by the NodeMinmaxEvaluation class
class NodeWithValue(TreeEvaluationChild[AnyTurnState], Protocol):
    """Represents a node with a value in a tree structure.

    Attributes:
        tree_evaluation (NodeMinmaxEvaluation): The minmax evaluation associated with the node.
        tree_node (TreeNode[Self]): The tree node associated with the node.

    Note: Uses Self to indicate that tree_node's children type should match the node itself.

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
    r"""Value-first minimax evaluation attached to a tree node."""

    _default_backup_policy_factory: ClassVar[BackupPolicyFactory | None] = (
        make_default_backup_policy
    )

    # objective responsible for semantic interpretation of Value objects at this node
    objective: Objective[StateT] | None = field(default_factory=make_default_objective)

    @property
    def minmax_value(self) -> Value | None:
        """Return this family's storage-name alias for generic backed_up_value."""
        return self.backed_up_value

    @minmax_value.setter
    def minmax_value(self, value: Value | None) -> None:
        """Set this family's storage-name alias for generic backed_up_value."""
        self.backed_up_value = value

    @property
    def branch_ordering_keys(self) -> dict[BranchKey, BranchOrderingKey]:
        """Return the cached generic branch-ordering keys for this node."""
        return self.decision_ordering.branch_ordering_keys

    @property
    def branches_sorted_by_value(self) -> dict[BranchKey, BranchOrderingKey]:
        """Compatibility alias for callers that still use the legacy name."""
        return self.branch_ordering_keys

    def is_over(self) -> bool:
        """Temporary compatibility alias for callers that still use legacy naming."""
        return self.is_terminal()

    def one_of_best_children_becomes_best_next_node(self) -> bool:
        """Refresh the PV head from the currently selected deterministic best child."""
        best_branch_key = self.best_branch()
        assert best_branch_key is not None
        has_best_branch_seq_changed = self.set_best_branch_sequence(
            self.best_branch_line_from_child(best_branch_key)
        )
        assert self.best_branch_sequence
        return has_best_branch_seq_changed
