"""Provide a factory function for creating node evaluators."""

from anemone._valanga_types import AnyTurnState

from .node_direct_evaluator import NodeDirectEvaluator
from .protocols import MasterStateValueEvaluator


def create_node_evaluator[StateT: AnyTurnState = AnyTurnState](
    master_state_evaluator: MasterStateValueEvaluator,
) -> NodeDirectEvaluator[StateT]:
    """Create a NodeDirectEvaluator backed by a master state evaluator."""
    return NodeDirectEvaluator(master_state_evaluator=master_state_evaluator)
