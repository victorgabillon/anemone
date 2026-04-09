"""Module for evaluating algorithm nodes directly using a master state evaluator."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain
from typing import TYPE_CHECKING, Any, Protocol, cast

from valanga.evaluations import Certainty, Value

from anemone._valanga_types import AnyOverEvent, AnyTurnState
from anemone.node_evaluation.common import canonical_value
from anemone.nodes.algorithm_node import AlgorithmNode

from .protocols import MasterStateValueEvaluator

if TYPE_CHECKING:
    from anemone.node_evaluation.tree.adversarial.node_minmax_evaluation import (
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


class EvaluationQueries[StateT: AnyTurnState = AnyTurnState]:
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


def _empty_algorithm_node_list[StateT: AnyTurnState](
) -> list[AlgorithmNode[StateT]]:
    """Return a new typed list of algorithm nodes."""
    return []


@dataclass(slots=True)
class DirectEvaluationOutcome[StateT: AnyTurnState = AnyTurnState]:
    """Outcome of directly evaluating a batch of nodes."""

    evaluated_nodes: list[AlgorithmNode[StateT]] = field(
        default_factory=_empty_algorithm_node_list
    )
    changed_nodes: list[AlgorithmNode[StateT]] = field(
        default_factory=_empty_algorithm_node_list
    )
    skipped_terminal_count: int = 0

    @property
    def reevaluated_count(self) -> int:
        """Return how many nodes were actually evaluated."""
        return len(self.evaluated_nodes)

    @property
    def changed_count(self) -> int:
        """Return how many nodes received a different direct value."""
        return len(self.changed_nodes)


class NodeDirectEvaluator[StateT: AnyTurnState = AnyTurnState]:
    """Evaluate the value of nodes in a tree structure.

    The evaluator uses a master state evaluator to calculate node values.
    """

    master_state_value_evaluator: MasterStateValueEvaluator
    current_evaluator_version: int

    def __init__(
        self,
        master_state_evaluator: MasterStateValueEvaluator,
    ) -> None:
        """Initialize a new instance of the NodeDirectEvaluator class."""
        self.master_state_value_evaluator = master_state_evaluator
        self.current_evaluator_version = 0

    def _store_direct_value(
        self,
        *,
        node: AlgorithmNode[StateT],
        value: Value,
    ) -> None:
        """Store one direct evaluation and stamp its evaluator provenance."""
        node.tree_evaluation.direct_value = value
        node.tree_evaluation.direct_evaluation_version = (
            self.current_evaluator_version
        )

    def _is_structurally_terminal_state(self, node: AlgorithmNode[StateT]) -> bool:
        """Return whether ``node`` is terminal by its own game-state semantics.

        Reevaluation skip logic must only exclude nodes whose *local state* is
        terminal. We intentionally do not use ``node.tree_evaluation.is_terminal()``
        here because canonical tree-evaluation state can become terminal/exact
        through backed-up child values on a non-terminal parent.
        """
        return node.state.is_game_over()

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
        self._store_direct_value(
            node=node,
            value=canonical_value.make_terminal_value(
                score=terminal_score,
                over_event=over_event,
            ),
        )
        assert node.tree_evaluation.is_terminal()

    def _terminal_score(
        self,
        *,
        node: AlgorithmNode[StateT],
        evaluation: float | None,
        canonical_over_event: AnyOverEvent,
    ) -> float:
        if evaluation is not None:
            return evaluation
        tree_eval: NodeMinmaxEvaluation[Any, Any] = cast("Any", node.tree_evaluation)
        return float(
            tree_eval.required_objective.terminal_score(
                canonical_over_event, node.state
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

        if direct_value is not None:
            evaluation_queries.over_nodes.append(node)
            return

        evaluation_queries.not_over_nodes.append(node)

    def evaluate_nodes(
        self,
        nodes: Sequence[AlgorithmNode[StateT]],
        *,
        evaluation_queries: EvaluationQueries[StateT],
        clear_existing_direct_values: bool,
        skip_terminal_nodes: bool,
    ) -> DirectEvaluationOutcome[StateT]:
        """Directly evaluate arbitrary nodes through the shared batch flow."""
        evaluation_queries.clear_queries()

        old_direct_values: dict[int, Value | None] = {}
        evaluated_nodes: list[AlgorithmNode[StateT]] = []
        skipped_terminal_count = 0
        seen_node_ids: set[int] = set()

        for node in nodes:
            node_identity = id(node)
            if node_identity in seen_node_ids:
                continue
            seen_node_ids.add(node_identity)

            if skip_terminal_nodes and self._is_structurally_terminal_state(node):
                skipped_terminal_count += 1
                continue

            old_direct_values[node_identity] = node.tree_evaluation.direct_value
            if clear_existing_direct_values:
                node.tree_evaluation.clear_direct_evaluation()

            self.add_evaluation_query(node, evaluation_queries)
            evaluated_nodes.append(node)

        if not evaluated_nodes:
            return DirectEvaluationOutcome(
                skipped_terminal_count=skipped_terminal_count
            )

        self.evaluate_all_queried_nodes(evaluation_queries=evaluation_queries)

        changed_nodes = [
            node
            for node in evaluated_nodes
            if old_direct_values[id(node)] != node.tree_evaluation.direct_value
        ]
        return DirectEvaluationOutcome(
            evaluated_nodes=evaluated_nodes,
            changed_nodes=changed_nodes,
            skipped_terminal_count=skipped_terminal_count,
        )

    def evaluate_all_not_over(
        self, not_over_nodes: list[AlgorithmNode[StateT]]
    ) -> None:
        """Evaluate all non-terminal nodes and store their evaluations."""
        values = self.master_state_value_evaluator.evaluate_batch_items(not_over_nodes)
        for node, value in zip(not_over_nodes, values, strict=True):
            processed_score = self.process_evalution_not_over(value.score, node)
            if value.certainty == Certainty.TERMINAL:
                raise DirectValueInvariantError(
                    node_id=node.tree_node.id,
                    reason=(
                        "non-terminal batch evaluation returned TERMINAL; "
                        "terminal certainty must come from node-local over detection"
                    ),
                )
            if value.certainty == Certainty.FORCED:
                processed_value = canonical_value.make_forced_value(
                    score=processed_score,
                    over_event=value.over_event,
                )
            else:
                if value.over_event is not None:
                    raise DirectValueInvariantError(
                        node_id=node.tree_node.id,
                        reason=(
                            "ESTIMATE direct evaluation must not carry over_event "
                            "metadata"
                        ),
                    )
                processed_value = canonical_value.make_estimate_value(
                    score=processed_score
                )
            self._store_direct_value(node=node, value=processed_value)
            assert node.tree_evaluation.direct_value is not None, (
                f"direct_value must be set for non-terminal node {node.tree_node.id}"
            )

    def process_evalution_not_over(
        self, evaluation: float, node: AlgorithmNode[StateT]
    ) -> float:
        """Process the evaluation for a node that is not over."""
        return (1 / DISCOUNT) ** node.tree_depth * evaluation
