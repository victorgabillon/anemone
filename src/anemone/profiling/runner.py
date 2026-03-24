"""Standalone execution shell for one profiling scenario run."""
# pylint: disable=broad-exception-caught,duplicate-code

from __future__ import annotations

import argparse
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .artifacts import (
    RunArtifacts,
    RunMetadata,
    RunResult,
    RunStatus,
    RunTimingSummary,
)
from .component_summary import (
    COMPONENT_SUMMARY_FILENAME,
    save_component_summary,
)
from .external_profilers import (
    SUPPORTED_EXTERNAL_PROFILERS,
    ProfilerName,
    run_with_external_profiler,
)
from .scenario_runtime import ScenarioRuntime, ScenarioRuntimeOptions
from .scenarios import get_scenario
from .storage import (
    default_runs_base_dir,
    make_run_dir,
    make_run_id,
    run_result_path,
    save_run_result,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _timestamp_utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(tz=UTC)


def _format_utc_timestamp(timestamp: datetime) -> str:
    """Format a UTC timestamp as a compact ISO string."""
    return (
        timestamp.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def _discover_git_commit(cwd: Path) -> str | None:
    """Best-effort lookup of the current git commit hash."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def run_scenario(
    scenario_name: str,
    output_dir: Path,
    *,
    command: Sequence[str] | None = None,
    cwd: Path | None = None,
    profiler: ProfilerName = "none",
    component_summary: bool = False,
    scenario_label: str | None = None,
) -> RunResult:
    """Execute one profiling scenario and persist its run artifact."""
    scenario = get_scenario(scenario_name)
    current_cwd = Path.cwd() if cwd is None else Path(cwd)
    started_at = _timestamp_utc_now()
    run_name = scenario.name if scenario_label is None else scenario_label
    requested_run_id = make_run_id(run_name, timestamp=started_at)
    run_dir = make_run_dir(Path(output_dir), requested_run_id)
    run_id = run_dir.name
    run_json_path = run_result_path(run_dir)
    metadata = RunMetadata(
        run_id=run_id,
        scenario_name=scenario.name,
        started_at_utc=_format_utc_timestamp(started_at),
        finished_at_utc=None,
        hostname=socket.gethostname() or None,
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_commit=_discover_git_commit(current_cwd),
        cwd=str(current_cwd.resolve()),
        command=[] if command is None else list(command),
        notes={
            "scenario_description": scenario.description,
            "profiler": profiler,
            "component_summary_requested": str(component_summary).lower(),
            "wall_time_definition": (
                "scenario execution wall time excluding profiler artifact writing"
            ),
        },
    )
    if scenario_label is not None:
        metadata.notes["scenario_label"] = scenario_label

    status = RunStatus.SUCCESS
    error_message: str | None = None
    extra_paths: dict[str, str] = {}
    scenario_runtime: ScenarioRuntime | None = None
    total_run_wall_time_seconds = 0.0
    try:
        if scenario.runtime_builder is None:
            scenario_runtime = ScenarioRuntime(runner=scenario.runner)
        else:
            scenario_runtime = scenario.runtime_builder(
                ScenarioRuntimeOptions(component_summary=component_summary)
            )
        metadata.notes["component_summary_available"] = str(
            scenario_runtime.component_collectors is not None
        ).lower()
        profiler_result = run_with_external_profiler(
            profiler,
            scenario_runtime.runner,
            run_dir,
        )
        total_run_wall_time_seconds = profiler_result.execution_wall_time_seconds
        extra_paths.update(profiler_result.extra_paths)
        if profiler_result.captured_error is not None:
            status = RunStatus.FAILED
            error_message = (
                "Scenario execution failed: "
                f"{type(profiler_result.captured_error).__name__}: "
                f"{profiler_result.captured_error}"
            )
    except Exception as exc:  # pragma: no cover - exercised by tests
        status = RunStatus.FAILED
        error_message = f"Scenario execution failed: {type(exc).__name__}: {exc}"

    if (
        component_summary
        and scenario_runtime is not None
        and scenario_runtime.component_collectors is not None
    ):
        try:
            component_summary_path = run_dir / COMPONENT_SUMMARY_FILENAME
            summary = scenario_runtime.component_collectors.build_summary(
                total_run_wall_time_seconds
            )
            save_component_summary(summary, component_summary_path)
            extra_paths["component_summary_json"] = str(component_summary_path)
            metadata.notes["component_summary_enabled"] = "true"
        except Exception as exc:  # pragma: no cover - simple artifact guard
            component_error = (
                f"Component summary generation failed: {type(exc).__name__}: {exc}"
            )
            if error_message is None:
                status = RunStatus.FAILED
                error_message = component_error
            else:
                error_message = f"{error_message}; {component_error}"

    finished_at = _timestamp_utc_now()
    result = RunResult(
        status=status,
        metadata=metadata,
        timing=RunTimingSummary(wall_time_seconds=total_run_wall_time_seconds),
        artifacts=RunArtifacts(
            run_json_path=str(run_json_path),
            extra_paths=extra_paths,
        ),
        error_message=error_message,
    )
    result.metadata.finished_at_utc = _format_utc_timestamp(finished_at)
    save_run_result(result, run_json_path)
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the lightweight runner entrypoint parser."""
    parser = argparse.ArgumentParser(
        prog="python -m anemone.profiling.runner",
        description="Run a single profiling scenario and write a run artifact.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Registered profiling scenario name to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_runs_base_dir()),
        help="Base directory where the profiling run folder will be created.",
    )
    parser.add_argument(
        "--profiler",
        choices=SUPPORTED_EXTERNAL_PROFILERS,
        default="none",
        help="Optional external profiler to run around the scenario.",
    )
    parser.add_argument(
        "--component-summary",
        action="store_true",
        help="Enable wrapper-based component timing when the scenario supports it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``python -m anemone.profiling.runner``."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_argument_parser()
    args = parser.parse_args(argv_list)

    try:
        result = run_scenario(
            args.scenario,
            Path(args.output_dir),
            command=[sys.executable, "-m", "anemone.profiling.runner", *argv_list],
            profiler=cast("ProfilerName", args.profiler),
            component_summary=args.component_summary,
        )
    except Exception as exc:  # pragma: no cover - simple CLI guard
        print(f"profiling run failed before artifact creation: {exc}", file=sys.stderr)
        return 1

    if result.status is RunStatus.SUCCESS:
        if result.artifacts.run_json_path is not None:
            print(result.artifacts.run_json_path)
        return 0

    if result.artifacts.run_json_path is not None:
        print(result.artifacts.run_json_path, file=sys.stderr)
    if result.error_message is not None:
        print(result.error_message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
