"""Both checkers must reject each marked invalid generic use."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valanga import BranchKey, State, Transition

    from anemone.dynamics import SearchDynamics, StatelessDynamicsAdapter
    from anemone.node_evaluation.tree.factory import NodeTreeMinmaxEvaluationFactory
    from anemone.nodes.state_handles import StateHandle
    from anemone.nodes.tree_node import TreeNode

    from .contracts import Position

bad_state: SearchDynamics[object, str]  # expect-error
bad_key: SearchDynamics[Position, list[int]]  # expect-error
bad_adapter: StatelessDynamicsAdapter[object]  # expect-error
bad_family: TreeNode[object, Position]  # expect-error
bad_minimax: NodeTreeMinmaxEvaluationFactory[Position]  # expect-error


def reversed_handle(handle: StateHandle[State]) -> StateHandle[Position]:
    """Reject a generic reader where a concrete state is promised."""
    return handle  # expect-error


def wrong_transition(
    dynamics: SearchDynamics[Position, BranchKey], state: Position
) -> Transition[str]:
    """Reject transitions with a different state payload."""
    return dynamics.step(state, "move", depth=0)  # expect-error
