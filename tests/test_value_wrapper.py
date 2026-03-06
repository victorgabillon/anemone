"""Tests for evaluator wrappers around Value objects."""

from anemone.node_evaluation.node_direct_evaluation.value_wrappers import (
    FloatToValueEvaluator,
)
from anemone.values import Certainty


class _NeverOver:
    def check_obvious_over_events(self, state):  # noqa: ANN001
        return None, None


class _Eval:
    over = _NeverOver()

    def evaluate(self, state) -> float:  # noqa: ANN001
        del state
        return 0.25


def test_float_to_value_evaluator_wraps_score() -> None:
    value = FloatToValueEvaluator(_Eval()).evaluate(object())
    assert value.score == 0.25
    assert value.certainty is Certainty.ESTIMATE
    assert value.over_event is None
