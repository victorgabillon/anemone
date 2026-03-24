"""Tests for suite-level profiling artifacts."""

from pathlib import Path

from anemone.profiling.artifacts import RunStatus
from anemone.profiling.suite_artifacts import (
    ScenarioAggregateSummary,
    ScenarioRepetitionSummary,
    SuiteRunResult,
    load_suite_run_result,
    save_suite_run_result,
)


def test_suite_run_result_round_trips_through_json(tmp_path: Path) -> None:
    """Suite artifacts should preserve key fields through JSON save/load."""
    result = SuiteRunResult(
        suite_name="baseline",
        description="Baseline deterministic profiling suite for Anemone.",
        run_id="2026-03-24T15-10-00_baseline_suite",
        started_at_utc="2026-03-24T15:10:00Z",
        finished_at_utc="2026-03-24T15:10:30Z",
        requested_repetitions=2,
        profiler="none",
        component_summary=True,
        scenario_aggregates=[
            ScenarioAggregateSummary(
                scenario_name="cheap_eval",
                requested_repetitions=2,
                completed_repetitions=2,
                successful_repetitions=2,
                failed_repetitions=0,
                wall_time_min_seconds=0.01,
                wall_time_max_seconds=0.02,
                wall_time_mean_seconds=0.015,
                wall_time_median_seconds=0.015,
                run_paths=["/tmp/rep1/run.json", "/tmp/rep2/run.json"],
            )
        ],
        scenario_runs=[
            ScenarioRepetitionSummary(
                scenario_name="cheap_eval",
                repetition_index=1,
                status=RunStatus.SUCCESS,
                run_json_path="/tmp/rep1/run.json",
                wall_time_seconds=0.01,
                component_summary_json_path="/tmp/rep1/component_summary.json",
            )
        ],
        notes={"scenario_runs_subdir": "scenario_runs"},
    )

    path = tmp_path / "suite.json"
    save_suite_run_result(result, path)
    loaded = load_suite_run_result(path)

    assert loaded.suite_name == result.suite_name
    assert loaded.run_id == result.run_id
    assert loaded.requested_repetitions == 2
    assert loaded.component_summary is True
    assert loaded.scenario_aggregates[0].scenario_name == "cheap_eval"
    assert loaded.scenario_runs[0].status is RunStatus.SUCCESS
    assert loaded.scenario_runs[0].component_summary_json_path is not None
