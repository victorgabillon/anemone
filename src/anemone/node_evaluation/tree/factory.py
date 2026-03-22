"""Provide factories for tree-evaluation families built on shared wiring."""

from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar

from valanga import State

from anemone._valanga_types import AnyTurnState
from anemone.backup_policies.protocols import BackupPolicy
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    NodeMinmaxEvaluation,
)
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    make_default_backup_policy as make_default_minmax_backup_policy,
)
from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
    make_default_objective as make_default_minmax_objective,
)
from anemone.node_evaluation.tree.node_tree_evaluation import NodeTreeEvaluation
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import Objective

StateT = TypeVar("StateT", bound=State)
TurnStateT = TypeVar("TurnStateT", bound=AnyTurnState)
EvalT_co = TypeVar("EvalT_co", bound=NodeTreeEvaluation[Any], covariant=True)
EvalT = TypeVar("EvalT", bound=NodeTreeEvaluation[Any])


class NodeTreeEvaluationFactory(Protocol[StateT]):
    """Factory protocol for generic tree-search node evaluations."""

    def create(
        self,
        tree_node: TreeNode[Any, StateT],
    ) -> NodeTreeEvaluation[StateT]:
        """Create a tree-search evaluation instance for the given node."""
        ...


class TreeEvaluationConstructor(Protocol[StateT, EvalT_co]):
    """Callable constructor shape shared by thin family evaluation wrappers."""

    def __call__(
        self,
        *,
        tree_node: TreeNode[Any, StateT],
        backup_policy: BackupPolicy[EvalT_co],
        objective: Objective[StateT],
    ) -> EvalT_co:
        """Create one concrete tree evaluation from shared assembled dependencies."""
        ...


def resolve_factory_dependency[DependencyT](
    dependency: DependencyT | None,
    *,
    default_factory: Callable[[], DependencyT],
) -> DependencyT:
    """Return the explicit dependency or lazily create the family default."""
    if dependency is not None:
        return dependency
    return default_factory()


class ConfiguredNodeTreeEvaluationFactory(
    NodeTreeEvaluationFactory[StateT],
    Generic[StateT, EvalT],  # noqa: UP046
):
    """Assemble one tree-evaluation family from shared backup/objective wiring."""

    backup_policy: BackupPolicy[EvalT]
    objective: Objective[StateT]

    def __init__(
        self,
        *,
        evaluation_type: TreeEvaluationConstructor[StateT, EvalT],
        backup_policy: BackupPolicy[EvalT],
        objective: Objective[StateT],
    ) -> None:
        """Store the shared construction recipe for one evaluation family."""
        self._evaluation_type = evaluation_type
        self.backup_policy = backup_policy
        self.objective = objective

    def create(
        self,
        tree_node: TreeNode[Any, StateT],
    ) -> EvalT:
        """Create one configured tree evaluation for the given tree node."""
        return self._evaluation_type(
            tree_node=tree_node,
            backup_policy=self.backup_policy,
            objective=self.objective,
        )


class NodeTreeMinmaxEvaluationFactory(
    ConfiguredNodeTreeEvaluationFactory[
        TurnStateT, NodeMinmaxEvaluation[Any, TurnStateT]
    ]
):
    """Create minimax evaluations by configuring the shared tree-evaluation assembly."""

    def __init__(
        self,
        backup_policy: BackupPolicy[NodeMinmaxEvaluation[Any, TurnStateT]]
        | None = None,
        objective: Objective[TurnStateT] | None = None,
    ) -> None:
        """Initialize the minimax factory with optional explicit dependencies."""
        super().__init__(
            evaluation_type=NodeMinmaxEvaluation,
            backup_policy=resolve_factory_dependency(
                backup_policy,
                default_factory=make_default_minmax_backup_policy,
            ),
            objective=resolve_factory_dependency(
                objective,
                default_factory=make_default_minmax_objective,
            ),
        )
