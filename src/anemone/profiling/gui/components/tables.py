"""Reusable table row builders for profiling runs and suites."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anemone.profiling.artifacts import RunResult
    from anemone.profiling.suite_artifacts import (
        ScenarioAggregateSummary,
        SuiteRunResult,
    )


def run_rows(runs: list[RunResult]) -> list[dict[str, object]]:
    """Build table rows for run browser views."""
    return [
        {
            "run_id": run.metadata.run_id,
            "scenario_name": run.metadata.scenario_name,
            "started_at_utc": run.metadata.started_at_utc,
            "wall_time_seconds": run.timing.wall_time_seconds,
            "status": run.status.value,
            "profiler": run.metadata.notes.get("profiler", "none"),
            "component_summary": run.metadata.notes.get(
                "component_summary_enabled",
                run.metadata.notes.get("component_summary_requested", "false"),
            ),
            "git_commit": run.metadata.git_commit or "",
        }
        for run in runs
    ]


def suite_rows(suites: list[SuiteRunResult]) -> list[dict[str, object]]:
    """Build table rows for suite browser views."""
    return [
        {
            "run_id": suite.run_id,
            "suite_name": suite.suite_name,
            "description": suite.description,
            "started_at_utc": suite.started_at_utc,
            "requested_repetitions": suite.requested_repetitions,
            "profiler": suite.profiler,
            "component_summary": suite.component_summary,
            "scenario_count": len(suite.scenario_aggregates),
        }
        for suite in suites
    ]


def suite_aggregate_rows(suite: SuiteRunResult) -> list[dict[str, object]]:
    """Build table rows for suite aggregate summaries."""
    return [
        {
            "scenario_name": aggregate.scenario_name,
            "requested_repetitions": aggregate.requested_repetitions,
            "successful_repetitions": aggregate.successful_repetitions,
            "failed_repetitions": aggregate.failed_repetitions,
            "success_rate": _success_rate(aggregate),
            "wall_time_mean_seconds": aggregate.wall_time_mean_seconds,
            "wall_time_median_seconds": aggregate.wall_time_median_seconds,
            "wall_time_min_seconds": aggregate.wall_time_min_seconds,
            "wall_time_max_seconds": aggregate.wall_time_max_seconds,
        }
        for aggregate in suite.scenario_aggregates
    ]


def repetition_rows(
    suite: SuiteRunResult,
    *,
    scenario_name: str | None = None,
) -> list[dict[str, object]]:
    """Build table rows for suite repetition records."""
    repetitions = suite.scenario_runs
    if scenario_name is not None:
        repetitions = [
            repetition
            for repetition in repetitions
            if repetition.scenario_name == scenario_name
        ]
    return [
        {
            "scenario_name": repetition.scenario_name,
            "repetition_index": repetition.repetition_index,
            "status": repetition.status.value,
            "wall_time_seconds": repetition.wall_time_seconds,
            "run_json_path": repetition.run_json_path or "",
            "component_summary_json_path": repetition.component_summary_json_path or "",
            "error_message": repetition.error_message or "",
        }
        for repetition in repetitions
    ]


def _success_rate(aggregate: ScenarioAggregateSummary) -> float:
    if aggregate.requested_repetitions == 0:
        return 0.0
    return aggregate.successful_repetitions / aggregate.requested_repetitions
