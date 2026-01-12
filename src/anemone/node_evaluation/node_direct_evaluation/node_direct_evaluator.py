"""
Module for evaluating algorithm nodes directly using a master state evaluator.
"""

from enum import Enum
from typing import Protocol, Sequence

from valanga import OverEvent, State
from valanga.evaluations import EvalItem

from anemone.nodes.algorithm_node import AlgorithmNode

DISCOUNT = 0.99999999  # lokks like at the moment the use is to break ties in the evaluation (not sure if needed or helpful now)


class NodeEvaluatorTypes(str, Enum):
    """
    Enum class representing different types of node evaluators.
    """

    NEURAL_NETWORK = "neural_network"


class NodeBatchValueEvaluator(Protocol):
    """Return value_white for each node, can use node.state_representation for speed."""

    def value_white_batch_from_nodes(
        self, nodes: Sequence[AlgorithmNode]
    ) -> list[float]:
        """Return value_white evaluations for a batch of nodes."""
        ...


class EvaluationQueries[StateT: State = State]:
    """
    A class that represents evaluation queries for algorithm nodes.

    Attributes:
        over_nodes (list[AlgorithmNode]): A list of algorithm nodes that are considered "over".
        not_over_nodes (list[AlgorithmNode]): A list of algorithm nodes that are not considered "over".
    """

    over_nodes: list[AlgorithmNode[StateT]]
    not_over_nodes: list[AlgorithmNode[StateT]]

    def __init__(self) -> None:
        """
        Initializes a new instance of the NodeEvaluator class.
        """
        self.over_nodes = []
        self.not_over_nodes = []

    def clear_queries(self) -> None:
        """
        Clears the evaluation queries by resetting the over_nodes and not_over_nodes lists.
        """
        self.over_nodes = []
        self.not_over_nodes = []


class OverEventDetector(Protocol):
    """Protocol for detecting over events in a game state."""

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        """Return an over event and evaluation if the state is terminal."""
        ...


class MasterStateEvaluator(Protocol):
    """Protocol for evaluating the value of a state."""

    over: OverEventDetector

    def value_white(self, state: State) -> float:
        """Evaluate a single state from white's perspective."""
        ...

    # the one method NodeEvaluator uses
    def value_white_batch_items[ItemStateT: State](
        self, items: Sequence[EvalItem[ItemStateT]]
    ) -> list[float]:
        """Evaluate a batch of items, defaulting to single-state calls."""
        # default fallback: single loop, state-only
        return [self.value_white(it.state) for it in items]


class NodeDirectEvaluator[StateT: State = State]:
    """
    The NodeEvaluator class is responsible for evaluating the value of nodes in a tree structure.
    It uses a board evaluator and a syzygy evaluator to calculate the value of the nodes.
    """

    master_state_evaluator: MasterStateEvaluator

    def __init__(
        self,
        master_state_evaluator: MasterStateEvaluator,
    ) -> None:
        """
        Initializes a new instance of the NodeEvaluator class.
        """
        self.master_state_evaluator = master_state_evaluator

    def check_obvious_over_events(self, node: AlgorithmNode[StateT]) -> None:
        """
        Updates the node.over object if the game is obviously over.
        """
        over_event: OverEvent | None
        evaluation: float | None
        over_event, evaluation = (
            self.master_state_evaluator.over.check_obvious_over_events(node.state)
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
            node.tree_evaluation.set_evaluation(evaluation=evaluation)

    def evaluate_all_queried_nodes(
        self, evaluation_queries: EvaluationQueries[StateT]
    ) -> None:
        """
        Evaluates all the queried nodes.
        """
        # node_over: AlgorithmNode
        # for node_over in evaluation_queries.over_nodes:
        # assert isinstance(node_over, AlgorithmNode)
        #    self.evaluate_over(node_over)

        if evaluation_queries.not_over_nodes:
            self.evaluate_all_not_over(evaluation_queries.not_over_nodes)

        evaluation_queries.clear_queries()

    def add_evaluation_query(
        self, node: AlgorithmNode[StateT], evaluation_queries: EvaluationQueries[StateT]
    ) -> None:
        """
        Adds an evaluation query for a node.
        """
        assert node.tree_evaluation.value_white_direct_evaluation is None
        self.check_obvious_over_events(node)
        if node.is_over():
            evaluation_queries.over_nodes.append(node)
        else:
            evaluation_queries.not_over_nodes.append(node)

    def evaluate_all_not_over(
        self, not_over_nodes: list[AlgorithmNode[StateT]]
    ) -> None:
        """Evaluate all non-terminal nodes and store their evaluations."""
        values = self.master_state_evaluator.value_white_batch_items(not_over_nodes)
        for node, v in zip(not_over_nodes, values, strict=True):
            node.tree_evaluation.set_evaluation(
                self.process_evalution_not_over(v, node)
            )

    def process_evalution_not_over(
        self, evaluation: float, node: AlgorithmNode[StateT]
    ) -> float:
        """
        Processes the evaluation for a node that is not over.
        """
        processed_evaluation = (1 / DISCOUNT) ** node.tree_depth * evaluation
        return processed_evaluation
