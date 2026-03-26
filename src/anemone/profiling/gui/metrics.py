"""Derived metrics used by the profiling dashboard."""

from __future__ import annotations

from statistics import stdev
from typing import TYPE_CHECKING, TypedDict

from anemone.profiling.artifacts import RunStatus

if TYPE_CHECKING:
    from anemone.profiling.component_summary import ComponentSummary, TimedCallStats
    from anemone.profiling.suite_artifacts import SuiteRunResult


class ComponentBreakdownRow(TypedDict):
    """High-level run component row for charts and metric cards."""

    component: str
    wall_time_seconds: float
    share_of_total: float


class ComponentDetailRow(TypedDict):
    """Detailed timed-component row for drill-down tables."""

    component: str
    wall_time_seconds: float
    share_of_total: float
    call_count: int | None
    mean_wall_time_seconds: float | None
    max_wall_time_seconds: float | None


class SuiteScenarioMetricRow(TypedDict):
    """Derived per-scenario aggregate metrics for one suite."""

    scenario_name: str
    successful_repetitions: int
    failed_repetitions: int
    success_rate: float
    mean_wall_time_seconds: float | None
    median_wall_time_seconds: float | None
    min_wall_time_seconds: float | None
    max_wall_time_seconds: float | None
    std_wall_time_seconds: float | None


class SuiteRepetitionMetricRow(TypedDict):
    """Successful per-repetition timing row for variability plots."""

    scenario_name: str
    repetition_index: int
    wall_time_seconds: float


class SuiteSummaryMetrics(TypedDict):
    """Compact suite-level summary metrics shown above the charts."""

    scenario_count: int
    mean_of_means_seconds: float | None
    fastest_scenario_name: str | None
    slowest_scenario_name: str | None


def component_breakdown_rows(summary: ComponentSummary) -> list[ComponentBreakdownRow]:
    """Return high-level run component rows for charts and metrics."""
    total_seconds = max(summary.total_run_wall_time_seconds, 0.0)
    evaluator_seconds = _timed_stat_seconds(summary.evaluator)
    dynamics_step_seconds = _timed_stat_seconds(summary.dynamics_step)
    dynamics_legal_actions_seconds = _timed_stat_seconds(summary.dynamics_legal_actions)
    dynamics_seconds = dynamics_step_seconds + dynamics_legal_actions_seconds
    overhead_seconds = _overhead_seconds(summary)

    return [
        _component_row(
            component="Evaluator",
            wall_time_seconds=evaluator_seconds,
            total_seconds=total_seconds,
        ),
        _component_row(
            component="Dynamics",
            wall_time_seconds=dynamics_seconds,
            total_seconds=total_seconds,
        ),
        _component_row(
            component="Overhead",
            wall_time_seconds=overhead_seconds,
            total_seconds=total_seconds,
        ),
    ]


def component_detail_rows(summary: ComponentSummary) -> list[ComponentDetailRow]:
    """Return lower-level component rows for detailed inspection."""
    total_seconds = max(summary.total_run_wall_time_seconds, 0.0)
    rows: list[ComponentDetailRow] = []

    rows.append(
        _timed_component_row(
            component="Evaluator",
            timed_stat=summary.evaluator,
            total_seconds=total_seconds,
        )
    )
    rows.append(
        _timed_component_row(
            component="Dynamics step",
            timed_stat=summary.dynamics_step,
            total_seconds=total_seconds,
        )
    )
    rows.append(
        _timed_component_row(
            component="Dynamics legal actions",
            timed_stat=summary.dynamics_legal_actions,
            total_seconds=total_seconds,
        )
    )
    rows.append(
        {
            "component": "Residual framework",
            "wall_time_seconds": _overhead_seconds(summary),
            "share_of_total": _ratio(
                _overhead_seconds(summary),
                total_seconds,
            ),
            "call_count": None,
            "mean_wall_time_seconds": None,
            "max_wall_time_seconds": None,
        }
    )
    return rows


def suite_scenario_metric_rows(suite: SuiteRunResult) -> list[SuiteScenarioMetricRow]:
    """Return suite scenario metrics with variability derived from repetitions."""
    successful_times_by_scenario = _successful_wall_times_by_scenario(suite)
    rows: list[SuiteScenarioMetricRow] = []
    for aggregate in suite.scenario_aggregates:
        successful_times = successful_times_by_scenario.get(aggregate.scenario_name, [])
        rows.append(
            {
                "scenario_name": aggregate.scenario_name,
                "successful_repetitions": aggregate.successful_repetitions,
                "failed_repetitions": aggregate.failed_repetitions,
                "success_rate": _ratio(
                    aggregate.successful_repetitions,
                    aggregate.requested_repetitions,
                ),
                "mean_wall_time_seconds": aggregate.wall_time_mean_seconds,
                "median_wall_time_seconds": aggregate.wall_time_median_seconds,
                "min_wall_time_seconds": aggregate.wall_time_min_seconds,
                "max_wall_time_seconds": aggregate.wall_time_max_seconds,
                "std_wall_time_seconds": _standard_deviation(successful_times),
            }
        )
    return sorted(
        rows,
        key=_scenario_mean_sort_key,
        reverse=True,
    )


def suite_repetition_metric_rows(
    suite: SuiteRunResult,
) -> list[SuiteRepetitionMetricRow]:
    """Return successful per-repetition timings for plotting variability."""
    rows: list[SuiteRepetitionMetricRow] = []
    for repetition in suite.scenario_runs:
        if (
            repetition.status is not RunStatus.SUCCESS
            or repetition.wall_time_seconds is None
        ):
            continue
        rows.append(
            {
                "scenario_name": repetition.scenario_name,
                "repetition_index": repetition.repetition_index,
                "wall_time_seconds": repetition.wall_time_seconds,
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["scenario_name"], row["repetition_index"]),
    )


def suite_summary_metrics(suite: SuiteRunResult) -> SuiteSummaryMetrics:
    """Return a compact summary of suite-wide performance metrics."""
    rows = suite_scenario_metric_rows(suite)
    if not rows:
        return {
            "scenario_count": 0,
            "mean_of_means_seconds": None,
            "fastest_scenario_name": None,
            "slowest_scenario_name": None,
        }

    mean_values = [
        row["mean_wall_time_seconds"]
        for row in rows
        if row["mean_wall_time_seconds"] is not None
    ]
    fastest_row = min(
        (row for row in rows if row["mean_wall_time_seconds"] is not None),
        key=lambda row: _require_float(row["mean_wall_time_seconds"]),
        default=None,
    )
    slowest_row = max(
        (row for row in rows if row["mean_wall_time_seconds"] is not None),
        key=lambda row: _require_float(row["mean_wall_time_seconds"]),
        default=None,
    )
    return {
        "scenario_count": len(rows),
        "mean_of_means_seconds": (
            None if not mean_values else sum(mean_values) / len(mean_values)
        ),
        "fastest_scenario_name": (
            None if fastest_row is None else fastest_row["scenario_name"]
        ),
        "slowest_scenario_name": (
            None if slowest_row is None else slowest_row["scenario_name"]
        ),
    }


def _component_row(
    *,
    component: str,
    wall_time_seconds: float,
    total_seconds: float,
) -> ComponentBreakdownRow:
    return {
        "component": component,
        "wall_time_seconds": wall_time_seconds,
        "share_of_total": _ratio(wall_time_seconds, total_seconds),
    }


def _timed_component_row(
    *,
    component: str,
    timed_stat: TimedCallStats | None,
    total_seconds: float,
) -> ComponentDetailRow:
    if timed_stat is None:
        return {
            "component": component,
            "wall_time_seconds": 0.0,
            "share_of_total": 0.0,
            "call_count": 0,
            "mean_wall_time_seconds": 0.0,
            "max_wall_time_seconds": 0.0,
        }

    total_wall_time_seconds = timed_stat.total_wall_time_seconds
    return {
        "component": component,
        "wall_time_seconds": total_wall_time_seconds,
        "share_of_total": _ratio(total_wall_time_seconds, total_seconds),
        "call_count": timed_stat.call_count,
        "mean_wall_time_seconds": timed_stat.mean_wall_time_seconds,
        "max_wall_time_seconds": timed_stat.max_wall_time_seconds,
    }


def _timed_stat_seconds(timed_stat: TimedCallStats | None) -> float:
    if timed_stat is None:
        return 0.0
    return timed_stat.total_wall_time_seconds


def _overhead_seconds(summary: ComponentSummary) -> float:
    residual = summary.residual_framework_wall_time_seconds
    if residual is not None:
        return max(float(residual), 0.0)
    return max(
        summary.total_run_wall_time_seconds
        - summary.total_profiled_component_wall_time_seconds,
        0.0,
    )


def _successful_wall_times_by_scenario(
    suite: SuiteRunResult,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for repetition in suite.scenario_runs:
        if (
            repetition.status is not RunStatus.SUCCESS
            or repetition.wall_time_seconds is None
        ):
            continue
        grouped.setdefault(repetition.scenario_name, []).append(
            repetition.wall_time_seconds
        )
    return grouped


def _standard_deviation(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return stdev(values)


def _require_float(value: float | None) -> float:
    if value is None:
        raise _missing_float_error()
    return value


def _scenario_mean_sort_key(row: SuiteScenarioMetricRow) -> float:
    return (
        0.0 if row["mean_wall_time_seconds"] is None else row["mean_wall_time_seconds"]
    )


def _missing_float_error() -> TypeError:
    return TypeError("Expected a float value, got None")


def _ratio(numerator: float | int, denominator: float | int) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        return 0.0
    return float(numerator) / denominator_value
