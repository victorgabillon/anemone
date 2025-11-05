"""
This module contains the implementation of the NodeEvaluator class, which is responsible for evaluating the value of
 nodes in a tree-based move selector.

The NodeEvaluator class wraps a board evaluator and a syzygy table to provide more complex evaluations of chess
 positions. It handles queries for evaluating nodes and manages obvious over events.

Classes:
- NodeEvaluator: Wrapping node evaluator with syzygy and obvious over event.

Enums:
- NodeEvaluatorTypes: Types of node evaluators.

Constants:
- DISCOUNT: Discount factor used in the evaluation.

Functions:
- None

"""

from enum import Enum
from typing import Protocol

from valanga import OverEvent, State

from anemone.basics import NodeState
from anemone.nodes.algorithm_node import AlgorithmNode

DISCOUNT = 0.99999999  # lokks like at the moment the use is to break ties in the evaluation (not sure if needed or helpful now)


class NodeEvaluatorTypes(str, Enum):
    """
    Enum class representing different types of node evaluators.
    """

    NEURAL_NETWORK = "neural_network"


class EvaluationQueries:
    """
    A class that represents evaluation queries for algorithm nodes.

    Attributes:
        over_nodes (list[AlgorithmNode]): A list of algorithm nodes that are considered "over".
        not_over_nodes (list[AlgorithmNode]): A list of algorithm nodes that are not considered "over".
    """

    over_nodes: list[AlgorithmNode]
    not_over_nodes: list[AlgorithmNode]

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


class MasterStateEvaluator(Protocol):
    """
    The MasterBoardEvaluator class is responsible for evaluating the value of chess positions (that are IBoard).
    It uses a board evaluator and a syzygy evaluator to calculate the value of the positions.
    """

    def value_white(self, state: State) -> float:
        """
        Calculates the value for the white player of a given node.
        If the value can be obtained from the syzygy evaluator, it is used.
        Otherwise, the board evaluator is used.
        """
        ...

    def check_obvious_over_events(
        self, state: State
    ) -> tuple[OverEvent | None, float | None]:
        """
        Checks if the given state is in an obvious game-over state and returns the corresponding OverEvent and evaluation.

        Args:
            board (boards.IBoard): The board to evaluate for game-over conditions.

        Raises:
            ValueError: If the board result string is not recognized.

        Returns:
            tuple[OverEvent | None, float]: A tuple containing the OverEvent
            (if the game is over or can be determined from Syzygy tables, otherwise None) and the evaluation score from White's perspective.
            The evaluation is especially useful when training models.
        """
        ...


class NodeEvaluator:
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
        Initializes a NodeEvaluator object.

        Args:
            state_evaluator (MasterStateEvaluator): The state evaluator used to evaluate the chess state.
        """
        self.master_state_evaluator = master_state_evaluator

    def check_obvious_over_events(self, node: AlgorithmNode) -> None:
        """
        Updates the node.over object if the game is obviously over.
        """
        over_event: OverEvent | None
        evaluation: float | None
        over_event, evaluation = (
            self.master_state_evaluator.check_obvious_over_events(node.state)
        )
        if over_event is not None:
            node.minmax_evaluation.over_event.becomes_over(
                how_over=over_event.how_over,
                who_is_winner=over_event.who_is_winner,
                termination=over_event.termination,
            )
            assert evaluation is not None, (
                "Evaluation should not be None for over nodes"
            )
            node.minmax_evaluation.set_evaluation(evaluation=evaluation)

    def evaluate_all_queried_nodes(self, evaluation_queries: EvaluationQueries) -> None:
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
        self, node: AlgorithmNode, evaluation_queries: EvaluationQueries
    ) -> None:
        """
        Adds an evaluation query for a node.
        """
        assert node.minmax_evaluation.value_white_evaluator is None
        self.check_obvious_over_events(node)
        if node.is_over():
            evaluation_queries.over_nodes.append(node)
        else:
            evaluation_queries.not_over_nodes.append(node)

    def evaluate_all_not_over(self, not_over_nodes: list[AlgorithmNode]) -> None:
        """
        Evaluates all the nodes that are not over.
        """

        node_not_over: AlgorithmNode
        for node_not_over in not_over_nodes:
            evaluation: float = self.master_state_evaluator.value_white(
                state=node_not_over.state
            )
            processed_evaluation: float = self.process_evalution_not_over(
                evaluation=evaluation, node=node_not_over
            )
            # assert isinstance(node_not_over, AlgorithmNode)
            node_not_over.minmax_evaluation.set_evaluation(processed_evaluation)

    def process_evalution_not_over(
        self, evaluation: float, node: AlgorithmNode
    ) -> float:
        """
        Processes the evaluation for a node that is not over.
        """
        processed_evaluation = (1 / DISCOUNT) ** node.tree_depth * evaluation
        return processed_evaluation
