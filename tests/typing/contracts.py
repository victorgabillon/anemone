"""Positive contracts for native recursive nodes, state handles and factories."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_type

from anemone.dynamics import SearchDynamics, StatelessDynamicsAdapter
from anemone.nodes.state_handles import MaterializedStateHandle, StateHandle

if TYPE_CHECKING:
    from valanga import BranchKey, Dynamics, State, Transition

    from anemone.node_evaluation.tree.factory import NodeTreeEvaluationFactory
    from anemone.node_evaluation.tree.node_tree_evaluation import NodeTreeEvaluation
    from anemone.nodes.itree_node import ITreeNode
    from anemone.nodes.tree_node import TreeNode


@dataclass(frozen=True)
class Position:
    """Small concrete state for structural and bound checking."""

    tag: int

    def is_game_over(self) -> bool:
        """Report terminal positions."""
        return self.tag == 0

    def pprint(self) -> str:
        """Render the state."""
        return str(self.tag)


def handle_covariance(handle: StateHandle[Position]) -> StateHandle[State]:
    """Read concrete states through the general State interface."""
    return handle


def node_state_reader(node: ITreeNode[Position]) -> Position:
    """Retain the state type while navigating a recursive node family."""
    return node.state


def defaults(handle: StateHandle) -> State:
    """Use the explicit State default when its parameter is omitted."""
    return handle.get()


def adaptation(dynamics: Dynamics[Position], state: Position) -> Transition[Position]:
    """Preserve state identity and the actual Valanga key contract."""
    adapter = StatelessDynamicsAdapter(dynamics)
    search: SearchDynamics[Position, BranchKey] = adapter
    assert_type(adapter, StatelessDynamicsAdapter[Position])
    return search.step(state, "move", depth=0)


def factory_relationship(
    factory: NodeTreeEvaluationFactory[Position],
    node: TreeNode[ITreeNode[Position], Position],
) -> NodeTreeEvaluation[Position]:
    """Couple the node state to its evaluation state."""
    return factory.create(node)


def materialized_default(state: Position) -> StateHandle[Position]:
    """Retain concrete payloads through native dataclass inference."""
    return MaterializedStateHandle(state)
