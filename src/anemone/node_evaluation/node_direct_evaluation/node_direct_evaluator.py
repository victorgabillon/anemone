"""Module for evaluating algorithm nodes directly using a master state evaluator."""

from collections.abc import Sequence
from enum import StrEnum
from itertools import chain
from typing import TYPE_CHECKING, Any, Protocol, cast

from valanga import OverEvent, TurnState

from anemone.nodes.algorithm_node import AlgorithmNode
from anemone.values import Certainty, Value

from .protocols import MasterStateValueEvaluator

if TYPE_CHECKING:
    from anemone.node_evaluation.node_tree_evaluation.node_minmax_evaluation import (
        NodeMinmaxEvaluation,
    )

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


class EvaluationQueries[StateT: TurnState = TurnState]:
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


class NodeDirectEvaluator[StateT: TurnState = TurnState]:
    """Evaluate the value of nodes in a tree structure.

    The evaluator uses a master state evaluator to calculate node values.
    """

    master_state_value_evaluator: MasterStateValueEvaluator

    def __init__(
        self,
        master_state_evaluator: MasterStateValueEvaluator,
    ) -> None:
        """Initialize a new instance of the NodeDirectEvaluator class."""
        self.master_state_value_evaluator = master_state_evaluator

    def check_obvious_over_events(self, node: AlgorithmNode[StateT]) -> None:
        """Check for obvious over events in the given node and update its evaluation accordingly."""
        over_event, evaluation = (
            self.master_state_value_evaluator.over.check_obvious_over_events(node.state)
        )
        if over_event is None:
            return

        assert over_event.is_over()
        terminal_score = self._terminal_score(
            node=node,
            evaluation=evaluation,
            canonical_over_event=over_event,
        )
        node.tree_evaluation.over_event = over_event
        node.tree_evaluation.direct_value = Value(
            score=terminal_score,
            certainty=Certainty.TERMINAL,
            over_event=over_event,
        )
        assert node.tree_evaluation.is_terminal_candidate()

    def _terminal_score(
        self,
        *,
        node: AlgorithmNode[StateT],
        evaluation: float | None,
        canonical_over_event: OverEvent,
    ) -> float:
        if evaluation is not None:
            return evaluation
        tree_eval: NodeMinmaxEvaluation[Any, Any] = cast("Any", node.tree_evaluation)
        return float(
            tree_eval.evaluation_ordering.terminal_score(
                canonical_over_event,
                perspective=node.state.turn,
            )
        )

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

        if node.tree_evaluation.is_terminal_candidate():
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
            assert node.tree_evaluation.direct_value is not None, (
                f"direct_value must be set for non-terminal node {node.tree_node.id}"
            )

    def process_evalution_not_over(
        self, evaluation: float, node: AlgorithmNode[StateT]
    ) -> float:
        """Process the evaluation for a node that is not over."""
        return (1 / DISCOUNT) ** node.tree_depth * evaluation
