"""Tests for timing wrappers around injectable components."""

from types import SimpleNamespace

from valanga import Color
from valanga.evaluations import Certainty, Value

from anemone.profiling.collectors import (
    DynamicsTimingCollector,
    EvaluatorTimingCollector,
    TimingMasterStateValueEvaluator,
    TimingSearchDynamics,
)


class _FakeEvaluator:
    def __init__(self) -> None:
        self.over = object()

    def evaluate(self, state: object) -> Value:
        return Value(score=self.value_white(state), certainty=Certainty.ESTIMATE)

    def evaluate_batch_items(self, items: list[object]) -> list[Value]:
        return [self.evaluate(getattr(item, "state", item)) for item in items]

    def value_white(self, state: object) -> float:
        return float(getattr(state, "score", 0.0))

    def value_white_batch_items(self, items: list[object]) -> list[float]:
        return [self.value_white(getattr(item, "state", item)) for item in items]


class _FakeDynamics:
    __anemone_search_dynamics__ = True

    def legal_actions(self, state: object) -> tuple[int, ...]:
        del state
        return (0, 1)

    def step(
        self, state: object, action: object, *, depth: int
    ) -> tuple[object, object, int]:
        return state, action, depth

    def action_name(self, state: object, action: object) -> str:
        del state
        return f"action:{action}"

    def action_from_name(self, state: object, name: str) -> int:
        del state
        return int(name.split(":")[1])


def test_timing_master_state_value_evaluator_preserves_results_and_counts() -> None:
    """The evaluator wrapper should preserve results while collecting timings."""
    collector = EvaluatorTimingCollector()
    wrapped = TimingMasterStateValueEvaluator(_FakeEvaluator(), collector)
    state = SimpleNamespace(score=2.5, turn=Color.WHITE)
    batch = [SimpleNamespace(state=state)]

    value = wrapped.evaluate(state)
    batch_values = wrapped.evaluate_batch_items(batch)
    white_value = wrapped.value_white(state)
    white_batch_values = wrapped.value_white_batch_items(batch)

    assert value.score == 2.5
    assert batch_values[0].score == 2.5
    assert white_value == 2.5
    assert white_batch_values == [2.5]
    assert collector.evaluate.call_count == 1
    assert collector.evaluate_batch_items.call_count == 1
    assert collector.value_white.call_count == 1
    assert collector.value_white_batch_items.call_count == 1
    assert collector.summary() is not None
    assert collector.summary().total_wall_time_seconds >= 0.0


def test_evaluator_summary_uses_non_overlapping_outermost_calls_only() -> None:
    """Nested evaluator timing records should not inflate the aggregate summary."""
    collector = EvaluatorTimingCollector()

    collector.record_timed_call(
        collector.evaluate,
        lambda: collector.record_timed_call(collector.value_white, lambda: None),
    )

    summary = collector.summary()

    assert summary is not None
    assert summary.call_count == 1
    assert collector.evaluate.call_count == 1
    assert collector.value_white.call_count == 1
    assert summary.total_wall_time_seconds >= 0.0


def test_timing_search_dynamics_preserves_results_and_counts() -> None:
    """The dynamics wrapper should preserve results while collecting timings."""
    collector = DynamicsTimingCollector()
    wrapped = TimingSearchDynamics(_FakeDynamics(), collector)
    state = SimpleNamespace(score=1.0)

    legal_actions = wrapped.legal_actions(state)
    transition = wrapped.step(state, 1, depth=3)

    assert legal_actions == (0, 1)
    assert transition == (state, 1, 3)
    assert wrapped.action_name(state, 1) == "action:1"
    assert wrapped.action_from_name(state, "action:1") == 1
    assert collector.legal_actions.call_count == 1
    assert collector.step.call_count == 1
    assert collector.legal_actions_summary() is not None
    assert collector.step_summary() is not None
