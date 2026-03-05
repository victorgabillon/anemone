"""Module for evaluating algorithm nodes directly using a master state evaluator."""

from collections.abc import Sequence
from enum import StrEnum
from itertools import chain
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, cast

from valanga import Color, OverEvent, State
from valanga.over_event import HowOver, Winner

from anemone.nodes.algorithm_node import AlgorithmNode
from anemone.values import Certainty, Value
from anemone.values.evaluation_ordering import TerminalOutcome

from .protocols import MasterStateEvaluator, MasterStateValueEvaluator
from .value_wrappers import FloatToValueEvaluator

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
        is_terminal = self._state_is_terminal(node)
        if over_event is None and not is_terminal:
            return

        canonical_over_event = self._canonical_terminal_over_event(
            raw_over_event=over_event,
            node=node,
        )
        terminal_score = self._terminal_score(
            node=node,
            evaluation=evaluation,
            canonical_over_event=canonical_over_event,
        )
        node.tree_evaluation.over_event = canonical_over_event
        node.tree_evaluation.direct_value = Value(
            score=terminal_score,
            certainty=Certainty.TERMINAL,
            over_event=canonical_over_event,
        )
        node.tree_evaluation.sync_float_views_from_values()
        assert node.tree_evaluation.direct_value is not None
        assert node.tree_evaluation.direct_value.certainty is Certainty.TERMINAL
        assert node.tree_evaluation.direct_value.over_event is not None
        assert node.tree_evaluation.direct_value.over_event.is_over()
        assert node.tree_evaluation.is_over()

    def _canonical_terminal_over_event(
        self,
        *,
        raw_over_event: OverEvent | None,
        node: AlgorithmNode[StateT],
    ) -> OverEvent:
        canonical = OverEvent()

        if self._is_usable_over_event(raw_over_event):
            canonical.becomes_over(
                how_over=raw_over_event.how_over,
                who_is_winner=raw_over_event.who_is_winner,
                termination=raw_over_event.termination,
            )
            if canonical.is_over():
                return canonical

        tree_eval: NodeMinmaxEvaluation[Any, Any] = cast("Any", node.tree_evaluation)
        terminal_outcome: TerminalOutcome = (
            tree_eval.evaluation_ordering.terminal_without_over_event
        )
        winner = self._winner_for_terminal_outcome(node=node, outcome=terminal_outcome)
        canonical = self._build_terminal_over_event(
            terminal_outcome=terminal_outcome,
            winner=winner,
        )
        assert canonical.is_over(), "Canonical terminal over_event must be over"
        return canonical

    def _is_usable_over_event(
        self, over_event: OverEvent | None
    ) -> TypeGuard[OverEvent]:
        if over_event is None:
            return False
        if not all(
            hasattr(over_event, attr)
            for attr in ("how_over", "who_is_winner", "termination")
        ):
            return False
        is_over_fn = getattr(over_event, "is_over", None)
        if callable(is_over_fn):
            return bool(is_over_fn())
        return True

    def _winner_for_terminal_outcome(
        self,
        *,
        node: AlgorithmNode[StateT],
        outcome: TerminalOutcome,
    ) -> Winner:
        turn = self._state_turn(node)
        if outcome is TerminalOutcome.DRAW:
            return Winner.NO_KNOWN_WINNER
        if outcome is TerminalOutcome.WIN:
            return self._winner_from_color(turn)
        winner_color = Color.BLACK if turn == Color.WHITE else Color.WHITE
        return self._winner_from_color(winner_color)

    def _state_turn(self, node: AlgorithmNode[StateT]) -> Color:
        turn_obj: object = getattr(node.state, "turn", None)
        assert isinstance(turn_obj, Color), (
            f"state.turn must be a valanga.Color, got {type(turn_obj)}"
        )
        return turn_obj

    def _state_base_score(self, node: AlgorithmNode[StateT]) -> float | None:
        if not hasattr(node.state, "base_score"):
            return None
        base: object = getattr(node.state, "base_score", None)
        if isinstance(base, (int, float)):
            return float(base)
        return None

    def _state_is_terminal(self, node: AlgorithmNode[StateT]) -> bool:
        return bool(getattr(node.state, "is_terminal", False))

    def _build_terminal_over_event(
        self,
        *,
        terminal_outcome: TerminalOutcome,
        winner: Winner,
    ) -> OverEvent:
        how_over = (
            HowOver.DRAW if terminal_outcome is TerminalOutcome.DRAW else HowOver.WIN
        )
        canonical = OverEvent()
        canonical.becomes_over(
            how_over=how_over,
            who_is_winner=winner,
            termination=None,
        )
        assert canonical.is_over(), (
            f"constructed OverEvent must be over: how_over={how_over} winner={winner}"
        )
        return canonical

    def _winner_from_color(self, color: Color) -> Winner:
        if color == Color.WHITE:
            return Winner.WHITE
        return Winner.BLACK

    def _terminal_score(
        self,
        *,
        node: AlgorithmNode[StateT],
        evaluation: float | None,
        canonical_over_event: OverEvent,
    ) -> float:
        if evaluation is not None:
            return evaluation
        base = self._state_base_score(node)
        if base is not None:
            return base
        tree_eval: NodeMinmaxEvaluation[Any, Any] = cast("Any", node.tree_evaluation)
        return float(
            tree_eval.evaluation_ordering.terminal_score(
                canonical_over_event,
                perspective=self._state_turn(node),
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
