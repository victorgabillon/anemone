"""Orchestrate repeatable profiling suite runs."""
# pylint: disable=broad-exception-caught,duplicate-code

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .artifacts import RunStatus
from .runner import run_scenario
from .storage import make_run_dir, make_run_id
from .suite_artifacts import (
    ScenarioAggregateSummary,
    ScenarioRepetitionSummary,
    SuiteRunResult,
    save_suite_run_result,
    suite_result_path,
)
from .suites import get_suite

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .external_profilers import ProfilerName


def _timestamp_utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(tz=UTC)


def _format_utc_timestamp(timestamp: datetime) -> str:
    """Format a UTC timestamp as a compact ISO string."""
    return (
        timestamp.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_suite(
    suite_name: str,
    output_dir: Path,
    *,
    repetitions: int,
    profiler: ProfilerName = "none",
    component_summary: bool = False,
    command: Sequence[str] | None = None,
) -> SuiteRunResult:
    """Execute a registered profiling suite and persist ``suite.json``."""
    if repetitions <= 0:
        raise _invalid_repetitions_error(repetitions)

    suite = get_suite(suite_name)
    started_at = _timestamp_utc_now()
    requested_run_id = make_run_id(f"{suite.name}_suite", timestamp=started_at)
    suite_dir = make_run_dir(Path(output_dir), requested_run_id)
    run_id = suite_dir.name
    scenario_runs_dir = suite_dir / "scenario_runs"
    scenario_runs_dir.mkdir(parents=True, exist_ok=True)

    scenario_runs: list[ScenarioRepetitionSummary] = []
    suite_error_message: str | None = None

    try:
        for scenario_name in suite.scenario_names:
            for repetition_index in range(1, repetitions + 1):
                scenario_label = f"{scenario_name}_rep{repetition_index}"
                try:
                    run_result = run_scenario(
                        scenario_name,
                        scenario_runs_dir,
                        command=command,
                        profiler=profiler,
                        component_summary=component_summary,
                        scenario_label=scenario_label,
                    )
                    scenario_runs.append(
                        ScenarioRepetitionSummary(
                            scenario_name=scenario_name,
                            repetition_index=repetition_index,
                            status=run_result.status,
                            run_json_path=run_result.artifacts.run_json_path,
                            wall_time_seconds=run_result.timing.wall_time_seconds,
                            component_summary_json_path=run_result.artifacts.extra_paths.get(
                                "component_summary_json"
                            ),
                            error_message=run_result.error_message,
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive suite guard
                    scenario_runs.append(
                        ScenarioRepetitionSummary(
                            scenario_name=scenario_name,
                            repetition_index=repetition_index,
                            status=RunStatus.FAILED,
                            run_json_path=None,
                            wall_time_seconds=None,
                            component_summary_json_path=None,
                            error_message=(
                                "Scenario execution failed before run artifact creation: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                    )
    except Exception as exc:  # pragma: no cover - defensive suite guard
        suite_error_message = f"Suite execution failed: {type(exc).__name__}: {exc}"

    scenario_aggregates = [
        ScenarioAggregateSummary.from_repetitions(
            scenario_name,
            [
                repetition
                for repetition in scenario_runs
                if repetition.scenario_name == scenario_name
            ],
            requested_repetitions=repetitions,
        )
        for scenario_name in suite.scenario_names
    ]

    finished_at = _timestamp_utc_now()
    result = SuiteRunResult(
        suite_name=suite.name,
        description=suite.description,
        run_id=run_id,
        started_at_utc=_format_utc_timestamp(started_at),
        finished_at_utc=_format_utc_timestamp(finished_at),
        requested_repetitions=repetitions,
        profiler=profiler,
        component_summary=component_summary,
        scenario_aggregates=scenario_aggregates,
        scenario_runs=scenario_runs,
        error_message=suite_error_message,
        notes={
            "scenario_runs_subdir": "scenario_runs",
            "scenario_count": str(len(suite.scenario_names)),
        },
    )
    save_suite_run_result(result, suite_result_path(suite_dir))
    return result


def _invalid_repetitions_error(repetitions: int) -> ValueError:
    return ValueError(f"repetitions must be at least 1, got {repetitions}")
