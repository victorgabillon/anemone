"""Provide a factory function for creating node evaluators."""

from valanga import State

from .node_direct_evaluator import MasterStateEvaluator, NodeDirectEvaluator


def create_node_evaluator[StateT: State = State](
    master_state_evaluator: MasterStateEvaluator,
) -> NodeDirectEvaluator[StateT]:
    """Create a NodeDirectEvaluator backed by a master state evaluator."""
    return NodeDirectEvaluator(master_state_evaluator=master_state_evaluator)
