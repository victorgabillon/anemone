"""Provide a factory function for creating node evaluators."""

from valanga import TurnState

from .node_direct_evaluator import NodeDirectEvaluator
from .protocols import MasterStateValueEvaluator


def create_node_evaluator[StateT: TurnState = TurnState](
    master_state_evaluator: MasterStateValueEvaluator,
) -> NodeDirectEvaluator[StateT]:
    """Create a NodeDirectEvaluator backed by a master state evaluator."""
    return NodeDirectEvaluator(master_state_evaluator=master_state_evaluator)
