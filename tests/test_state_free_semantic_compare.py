"""Exact objective parity and lazy-state coverage for candidate comparisons."""

from itertools import product
from types import SimpleNamespace
from typing import Any

import pytest
from valanga import Color, Outcome, OverEvent
from valanga.evaluations import Certainty, Value

from anemone.node_evaluation.common.value_candidate import ValueCandidateSource
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import SingleAgentMaxObjective

_SCORES = [float("-inf"), -3.5, -0.0, 0.0, 2.0, float("inf"), float("nan")]
_EVENTS = [None, *(OverEvent(outcome=outcome) for outcome in Outcome)]
_LINES = [None, [], [0, 1]]


@pytest.mark.parametrize("left_score,right_score", list(product(_SCORES, repeat=2)))
@pytest.mark.parametrize(
    "left_certainty,right_certainty", list(product(Certainty, repeat=2))
)
def test_single_agent_state_free_semantic_parity(
    left_score: float,
    right_score: float,
    left_certainty: Certainty,
    right_certainty: Certainty,
) -> None:
    """Compare every metadata pair, including metadata ignored by the objective."""
    objective = SingleAgentMaxObjective()
    states = [
        object(),
        SimpleNamespace(turn=Color.WHITE),
        SimpleNamespace(turn=Color.BLACK),
    ]
    for left_event, right_event, left_line, right_line in product(
        _EVENTS, _EVENTS, _LINES, _LINES
    ):
        left = Value(left_score, left_certainty, left_event, left_line)
        right = Value(right_score, right_certainty, right_event, right_line)
        actual = objective.semantic_compare_without_state(left, right)
        for state in states:
            assert actual == objective.semantic_compare(left, right, state)


class _CountingHandle:
    """Expose state requests and optionally reject every materialization."""

    def __init__(self, state: Any = None) -> None:
        self.state = state
        self.calls = 0

    def get(self) -> Any:
        """Return the supplied state, or fail if this is a lazy-state guard."""
        self.calls += 1
        assert self.state is not None, "StateHandle.get() must stay lazy"
        return self.state


class _ExplodingStateNode(TreeNode[Any, Any]):
    """Reject the property access itself, before the handle could be called."""

    @property
    def state(self) -> Any:
        pytest.fail("tree_node.state must stay lazy")


class _StatefulObjective:
    """Prefer high or low scores according to the actual node state."""

    def evaluate_value(self, value: Value, state: Any) -> float:
        """Project from this node's perspective."""
        return state.direction * value.score

    def semantic_compare(self, left: Value, right: Value, state: Any) -> int:
        """Deliberately require state for the ordering decision."""
        return state.direction * (
            (left.score > right.score) - (left.score < right.score)
        )

    def terminal_score(self, over_event: Any, state: Any) -> float:
        """Provide the remaining required objective method."""
        del over_event, state
        return 0.0


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("non_callable_hook", [False, True])
def test_stateful_semantic_fallback(direction: int, non_callable_hook: bool) -> None:
    """Absent and non-callable optional hooks preserve the stateful behavior."""
    objective = _StatefulObjective()
    assert not hasattr(objective, "semantic_compare_without_state")
    if non_callable_hook:
        objective.semantic_compare_without_state = None
    state = SimpleNamespace(direction=direction)
    handle = _CountingHandle(state)
    node = TreeNode(id_=0, tree_depth_=0, state_handle_=handle, parent_nodes_=None)
    evaluation = NodeMaxEvaluation(tree_node=node, objective=objective)
    direct = Value(2.0, Certainty.ESTIMATE)
    tree = Value(1.0, Certainty.ESTIMATE)
    evaluation.direct_value = direct
    evaluation.backed_up_value = tree

    assert (
        evaluation.compare_candidate_values(direct, tree)
        == objective.semantic_compare(direct, tree, state)
        == direction
    )
    assert handle.calls == 1
    candidate = evaluation.get_effective_value_candidate()
    assert candidate.value is (direct if direction == 1 else tree)
    assert handle.calls == 2
    assert not evaluation.is_terminal()
    assert handle.calls == 3


@pytest.mark.parametrize("guard", ["property", "handle"])
@pytest.mark.parametrize("certainty", Certainty)
@pytest.mark.parametrize(
    "direct_score,tree_score", [(2.0, 1.0), (1.0, 2.0), (1.0, 1.0)]
)
def test_public_candidate_and_terminality_paths_do_not_access_state(
    guard: str, certainty: Certainty, direct_score: float, tree_score: float
) -> None:
    """Exercise real candidate comparison, effective values, and is_terminal()."""
    handle = _CountingHandle()
    node_type = _ExplodingStateNode if guard == "property" else TreeNode
    node = node_type(id_=0, tree_depth_=0, state_handle_=handle, parent_nodes_=None)
    objective = SingleAgentMaxObjective()
    evaluation = NodeMaxEvaluation(tree_node=node, objective=objective)
    event = None if certainty == Certainty.ESTIMATE else OverEvent(outcome=Outcome.DRAW)
    direct = Value(direct_score, certainty, event)
    tree = Value(tree_score, certainty, event)
    evaluation.direct_value = direct
    evaluation.backed_up_value = tree
    expected = direct if direct_score > tree_score else tree
    expected_source = (
        ValueCandidateSource.DIRECT_SELF
        if direct_score > tree_score
        else ValueCandidateSource.TREE_CHILD
    )

    assert evaluation.compare_candidate_values(
        direct, tree
    ) == objective.semantic_compare(direct, tree, object())
    candidate = evaluation.get_effective_value_candidate()
    assert candidate.value is expected
    assert candidate.source == expected_source
    assert evaluation.effective_value is expected
    assert evaluation.get_value_candidate() is expected
    assert evaluation.is_terminal() == (certainty == Certainty.TERMINAL)
    assert evaluation.has_exact_value() == (certainty != Certainty.ESTIMATE)
    assert handle.calls == 0
