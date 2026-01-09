"""This module provides a factory function for creating node evaluators."""

from valanga import State

from .node_direct_evaluator import MasterStateEvaluator, NodeDirectEvaluator


def create_node_evaluator[StateT: State = State](
    master_state_evaluator: MasterStateEvaluator,
) -> NodeDirectEvaluator[StateT]:
    return NodeDirectEvaluator(master_state_evaluator=master_state_evaluator)
