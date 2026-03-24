"""Transparent timing wrappers for injectable profiling boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from anemone.dynamics import SearchDynamics
from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
    OverEventDetector,
)

from .component_summary import ComponentSummary, TimedCallStats

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import valanga
    from valanga import State


@dataclass(slots=True)
class _MutableTimingAccumulator:
    """Mutable accumulator for repeated wall-time measurements."""

    call_count: int = 0
    total_wall_time_seconds: float = 0.0
    max_wall_time_seconds: float = 0.0
    min_wall_time_seconds: float | None = None

    def record(self, duration: float) -> None:
        """Record one measured duration."""
        self.call_count += 1
        self.total_wall_time_seconds += duration
        self.max_wall_time_seconds = max(self.max_wall_time_seconds, duration)
        if self.min_wall_time_seconds is None:
            self.min_wall_time_seconds = duration
        else:
            self.min_wall_time_seconds = min(self.min_wall_time_seconds, duration)

    def freeze(self) -> TimedCallStats:
        """Convert the mutable accumulator into stable summary data."""
        mean_wall_time_seconds = (
            0.0
            if self.call_count == 0
            else self.total_wall_time_seconds / self.call_count
        )
        return TimedCallStats(
            call_count=self.call_count,
            total_wall_time_seconds=self.total_wall_time_seconds,
            max_wall_time_seconds=self.max_wall_time_seconds,
            min_wall_time_seconds=self.min_wall_time_seconds,
            mean_wall_time_seconds=mean_wall_time_seconds,
        )


def _record_timed_call(
    accumulator: _MutableTimingAccumulator,
    callback: Callable[[], Any],
) -> Any:
    """Execute one callback while recording its wall-clock duration."""
    started_at = time.perf_counter()
    try:
        return callback()
    finally:
        accumulator.record(time.perf_counter() - started_at)


@dataclass(slots=True)
class EvaluatorTimingCollector:
    """Method-level accumulators for a wrapped evaluator."""

    total: _MutableTimingAccumulator = field(default_factory=_MutableTimingAccumulator)
    evaluate: _MutableTimingAccumulator = field(
        default_factory=_MutableTimingAccumulator
    )
    evaluate_batch_items: _MutableTimingAccumulator = field(
        default_factory=_MutableTimingAccumulator
    )
    value_white: _MutableTimingAccumulator = field(
        default_factory=_MutableTimingAccumulator
    )
    value_white_batch_items: _MutableTimingAccumulator = field(
        default_factory=_MutableTimingAccumulator
    )
    _call_depth: int = 0

    def record_timed_call(
        self,
        accumulator: _MutableTimingAccumulator,
        callback: Callable[[], Any],
    ) -> Any:
        """Record one evaluator call without double-counting nested wrapper calls."""
        self._call_depth += 1
        started_at = time.perf_counter()
        try:
            return callback()
        finally:
            duration = time.perf_counter() - started_at
            accumulator.record(duration)
            if self._call_depth == 1:
                self.total.record(duration)
            self._call_depth -= 1

    def summary(self) -> TimedCallStats | None:
        """Return the aggregate evaluator timing summary."""
        return None if self.total.call_count == 0 else self.total.freeze()


@dataclass(slots=True)
class DynamicsTimingCollector:
    """Method-level accumulators for a wrapped dynamics object."""

    legal_actions: _MutableTimingAccumulator = field(
        default_factory=_MutableTimingAccumulator
    )
    step: _MutableTimingAccumulator = field(default_factory=_MutableTimingAccumulator)

    def step_summary(self) -> TimedCallStats | None:
        """Return the timing summary for dynamics `step(...)`."""
        return None if self.step.call_count == 0 else self.step.freeze()

    def legal_actions_summary(self) -> TimedCallStats | None:
        """Return the timing summary for dynamics `legal_actions(...)`."""
        return (
            None if self.legal_actions.call_count == 0 else self.legal_actions.freeze()
        )


@dataclass(slots=True)
class ComponentCollectors:
    """Bundle of collector state used to build a component summary artifact."""

    evaluator: EvaluatorTimingCollector | None = None
    dynamics: DynamicsTimingCollector | None = None

    def build_summary(self, total_run_wall_time_seconds: float) -> ComponentSummary:
        """Build the stable component-summary artifact for one run."""
        evaluator_summary = None if self.evaluator is None else self.evaluator.summary()
        dynamics_step_summary = (
            None if self.dynamics is None else self.dynamics.step_summary()
        )
        dynamics_legal_actions_summary = (
            None if self.dynamics is None else self.dynamics.legal_actions_summary()
        )

        total_profiled_component_wall_time_seconds = sum(
            stat.total_wall_time_seconds
            for stat in [
                evaluator_summary,
                dynamics_step_summary,
                dynamics_legal_actions_summary,
            ]
            if stat is not None
        )
        residual_framework_wall_time_seconds = max(
            total_run_wall_time_seconds - total_profiled_component_wall_time_seconds,
            0.0,
        )

        return ComponentSummary(
            total_run_wall_time_seconds=total_run_wall_time_seconds,
            total_profiled_component_wall_time_seconds=(
                total_profiled_component_wall_time_seconds
            ),
            residual_framework_wall_time_seconds=residual_framework_wall_time_seconds,
            evaluator=evaluator_summary,
            dynamics_step=dynamics_step_summary,
            dynamics_legal_actions=dynamics_legal_actions_summary,
            notes={
                "residual_definition": (
                    "total_run_wall_time - wrapped_component_wall_times"
                )
            },
        )


class TimingMasterStateValueEvaluator(MasterStateValueEvaluator):
    """Transparent timing wrapper around a master state evaluator."""

    over: OverEventDetector

    def __init__(
        self,
        wrapped: MasterStateValueEvaluator,
        collector: EvaluatorTimingCollector,
    ) -> None:
        """Wrap one evaluator while preserving its public protocol."""
        self._wrapped = wrapped
        self.collector = collector
        self.over = wrapped.over

    def evaluate(self, state: State) -> Any:
        """Time `evaluate(...)` while preserving semantics."""
        return self.collector.record_timed_call(
            self.collector.evaluate,
            lambda: self._wrapped.evaluate(state),
        )

    def evaluate_batch_items(
        self,
        items: Sequence[Any],
    ) -> list[Any]:
        """Time `evaluate_batch_items(...)` while preserving semantics."""
        return cast(
            "list[Any]",
            self.collector.record_timed_call(
                self.collector.evaluate_batch_items,
                lambda: self._wrapped.evaluate_batch_items(items),
            ),
        )

    def value_white(self, state: State) -> float:
        """Time `value_white(...)` when the wrapped evaluator provides it."""
        return cast(
            "float",
            self.collector.record_timed_call(
                self.collector.value_white,
                lambda: self._wrapped.value_white(state),  # type: ignore[attr-defined]
            ),
        )

    def value_white_batch_items(self, items: Sequence[Any]) -> list[float]:
        """Time `value_white_batch_items(...)` when the evaluator provides it."""
        return cast(
            "list[float]",
            self.collector.record_timed_call(
                self.collector.value_white_batch_items,
                lambda: self._wrapped.value_white_batch_items(items),  # type: ignore[attr-defined]
            ),
        )

    def __getattr__(self, name: str) -> object:
        """Delegate all other attribute access to the wrapped evaluator."""
        return getattr(self._wrapped, name)


class TimingSearchDynamics(SearchDynamics[Any, Any]):
    """Transparent timing wrapper around a search dynamics object."""

    __anemone_search_dynamics__ = True

    def __init__(
        self,
        wrapped: SearchDynamics[Any, Any],
        collector: DynamicsTimingCollector,
    ) -> None:
        """Wrap one dynamics object while preserving its public protocol."""
        self._wrapped = wrapped
        self.collector = collector

    def legal_actions(self, state: Any) -> valanga.BranchKeyGeneratorP[Any]:
        """Time `legal_actions(...)` while preserving semantics."""
        return cast(
            "valanga.BranchKeyGeneratorP[Any]",
            _record_timed_call(
                self.collector.legal_actions,
                lambda: self._wrapped.legal_actions(state),
            ),
        )

    def step(self, state: Any, action: Any, *, depth: int) -> valanga.Transition[Any]:
        """Time `step(...)` while preserving semantics."""
        return cast(
            "valanga.Transition[Any]",
            _record_timed_call(
                self.collector.step,
                lambda: self._wrapped.step(state, action, depth=depth),
            ),
        )

    def action_name(self, state: Any, action: Any) -> str:
        """Delegate `action_name(...)` to the wrapped dynamics."""
        return self._wrapped.action_name(state, action)

    def action_from_name(self, state: Any, name: str) -> Any:
        """Delegate `action_from_name(...)` to the wrapped dynamics."""
        return self._wrapped.action_from_name(state, name)

    def __getattr__(self, name: str) -> object:
        """Delegate all other attribute access to the wrapped dynamics."""
        return getattr(self._wrapped, name)


def wrap_profiled_components(
    *,
    evaluator: MasterStateValueEvaluator | None,
    dynamics: SearchDynamics[Any, Any] | None,
) -> tuple[
    MasterStateValueEvaluator | None,
    SearchDynamics[Any, Any] | None,
    ComponentCollectors,
]:
    """Wrap supported injectable components and return their collectors."""
    collectors = ComponentCollectors()

    wrapped_evaluator = evaluator
    if evaluator is not None:
        collectors.evaluator = EvaluatorTimingCollector()
        wrapped_evaluator = TimingMasterStateValueEvaluator(
            evaluator,
            collectors.evaluator,
        )

    wrapped_dynamics = dynamics
    if dynamics is not None:
        collectors.dynamics = DynamicsTimingCollector()
        wrapped_dynamics = TimingSearchDynamics(dynamics, collectors.dynamics)

    return wrapped_evaluator, wrapped_dynamics, collectors
