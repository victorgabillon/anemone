"""Module for evaluating algorithm nodes directly using a master state evaluator."""

from collections.abc import Sequence
from enum import StrEnum
from itertools import chain
from typing import Protocol

from valanga import OverEvent, State

from anemone.nodes.algorithm_node import AlgorithmNode
from anemone.values import Certainty, Value

from .protocols import MasterStateEvaluator, MasterStateValueEvaluator
from .value_wrappers import FloatToValueEvaluator

DISCOUNT = 0.99999999  # lokks like at the moment the use is to break ties in the evaluation (not sure if needed or helpful now)


class DirectValueInvariantError(AssertionError):
    """Custom error for invariant violations related to direct_value in NodeDirectEvaluator."""

    def __init__(self, *, node_id: int, reason: str) -> None:
        """Initialize the DirectValueInvariantError with a specific message."""
        super().__init__(f"direct_value invariant failed for node {node_id}: {reason}")


class NodeEvaluatorTypes(StrEnum):
    """Enum class representing different types of node evaluators."""

    NEURAL_NETWORK = "neural_network"


class NodeBatchValueEvaluator(Protocol):
    """Return value_white for each node, can use node.state_representation for speed."""

    def value_white_batch_from_nodes(
        self, nodes: Sequence[AlgorithmNode]
    ) -> list[float]:
        """Return value_white evaluations for a batch of nodes."""
        ...


class EvaluationQueries[StateT: State = State]:
    """A class that represents evaluation queries for algorithm nodes.

    Attributes:
        over_nodes (list[AlgorithmNode]): A list of algorithm nodes that are considered "over".
        not_over_nodes (list[AlgorithmNode]): A list of algorithm nodes that are not considered "over".

    """

    over_nodes: list[AlgorithmNode[StateT]]
    not_over_nodes: list[AlgorithmNode[StateT]]

    def __init__(self) -> None:
        """Initialize a new instance of the EvaluationQueries class."""
        self.over_nodes = []
        self.not_over_nodes = []

    def clear_queries(self) -> None:
        """Clear the evaluation queries by resetting the over_nodes and not_over_nodes lists."""
        self.over_nodes = []
        self.not_over_nodes = []


class NodeDirectEvaluator[StateT: State = State]:
    """Evaluate the value of nodes in a tree structure.

    The evaluator uses a master state evaluator to calculate node values.
    """

    master_state_value_evaluator: MasterStateValueEvaluator

    def __init__(
        self,
        master_state_evaluator: MasterStateEvaluator | MasterStateValueEvaluator,
    ) -> None:
        """Initialize a new instance of the NodeDirectEvaluator class."""
        self.master_state_value_evaluator = self._as_value_evaluator(
            master_state_evaluator
        )

    def _as_value_evaluator(
        self,
        evaluator: MasterStateEvaluator | MasterStateValueEvaluator,
    ) -> MasterStateValueEvaluator:
        """Normalize legacy float evaluators into Value evaluators."""
        if isinstance(evaluator, MasterStateValueEvaluator):
            return evaluator
        return FloatToValueEvaluator(inner=evaluator, _over=evaluator.over)

    def check_obvious_over_events(self, node: AlgorithmNode[StateT]) -> None:
        """Update the node.over object if the game is obviously over."""
        over_event: OverEvent | None
        evaluation: float | None
        over_event, evaluation = (
            self.master_state_value_evaluator.over.check_obvious_over_events(node.state)
        )
        if over_event is not None:
            node.tree_evaluation.over_event.becomes_over(
                how_over=over_event.how_over,
                who_is_winner=over_event.who_is_winner,
                termination=over_event.termination,
            )
            assert evaluation is not None, (
                "Evaluation should not be None for over nodes"
            )
            value = Value(
                score=evaluation,
                certainty=Certainty.TERMINAL,
                over_event=over_event,
            )
            node.tree_evaluation.direct_value = value
            node.tree_evaluation.sync_float_views_from_values()

    def evaluate_all_queried_nodes(
        self, evaluation_queries: EvaluationQueries[StateT]
    ) -> None:
        """Evaluate all the queried nodes."""
        if evaluation_queries.not_over_nodes:
            self.evaluate_all_not_over(evaluation_queries.not_over_nodes)

        for node in chain(
            evaluation_queries.over_nodes, evaluation_queries.not_over_nodes
        ):
            assert node.tree_evaluation.direct_value is not None, (
                f"direct_value must be set after evaluation for node {node.tree_node.id}"
            )

        evaluation_queries.clear_queries()

    def add_evaluation_query(
        self, node: AlgorithmNode[StateT], evaluation_queries: EvaluationQueries[StateT]
    ) -> None:
        """Add an evaluation query for a node."""
        if node.tree_evaluation.direct_value is not None:
            raise DirectValueInvariantError(
                node_id=node.tree_node.id,
                reason="direct_value must be None before add_evaluation_query",
            )

        self.check_obvious_over_events(node)

        # Re-read into a local after the call: this breaks mypy's earlier narrowing.
        direct_value: Value | None = node.tree_evaluation.direct_value

        if node.is_over():
            if direct_value is None:
                raise DirectValueInvariantError(
                    node_id=node.tree_node.id,
                    reason="direct_value must be set for terminal nodes after check_obvious_over_events",
                )
            evaluation_queries.over_nodes.append(node)
            return

        evaluation_queries.not_over_nodes.append(node)

    def evaluate_all_not_over(
        self, not_over_nodes: list[AlgorithmNode[StateT]]
    ) -> None:
        """Evaluate all non-terminal nodes and store their evaluations."""
        values = self.master_state_value_evaluator.evaluate_batch_items(not_over_nodes)
        for node, value in zip(not_over_nodes, values, strict=True):
            processed_value = Value(
                score=self.process_evalution_not_over(value.score, node),
                certainty=value.certainty,
                over_event=value.over_event,
            )
            node.tree_evaluation.direct_value = processed_value
            node.tree_evaluation.sync_float_views_from_values()
            assert node.tree_evaluation.direct_value is not None, (
                f"direct_value must be set for non-terminal node {node.tree_node.id}"
            )

    def process_evalution_not_over(
        self, evaluation: float, node: AlgorithmNode[StateT]
    ) -> float:
        """Process the evaluation for a node that is not over."""
        return (1 / DISCOUNT) ** node.tree_depth * evaluation
