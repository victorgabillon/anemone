"""Tests for profiling artifact serialization helpers."""

import json
from pathlib import Path

from anemone.profiling.artifacts import (
    RunArtifacts,
    RunMetadata,
    RunResult,
    RunStatus,
    RunTimingSummary,
    run_result_to_dict,
)
from anemone.profiling.storage import load_run_result, save_run_result


def test_run_result_round_trip_preserves_fields(tmp_path: Path) -> None:
    """Run artifacts should survive a save/load round trip."""
    run_result = RunResult(
        status=RunStatus.SUCCESS,
        metadata=RunMetadata(
            run_id="2026-03-24T14-32-10_smoke",
            scenario_name="smoke",
            started_at_utc="2026-03-24T14:32:10Z",
            finished_at_utc="2026-03-24T14:32:11Z",
            hostname="test-host",
            python_version="3.13.0",
            platform="Linux-test",
            git_commit="abc123",
            cwd="/repo",
            command=["python", "-m", "anemone.profiling.cli", "run"],
            notes={"scenario_description": "tiny scenario"},
        ),
        timing=RunTimingSummary(wall_time_seconds=0.125),
        artifacts=RunArtifacts(
            run_json_path="profiling_runs/2026-03-24T14-32-10_smoke/run.json",
            extra_paths={"future_report": "report.html"},
        ),
        error_message=None,
    )
    path = tmp_path / "run.json"

    save_run_result(run_result, path)
    loaded = load_run_result(path)

    assert loaded == run_result


def test_run_result_to_dict_exposes_stable_top_level_schema_keys() -> None:
    """The serialized schema should keep the expected top-level contract."""
    run_result = RunResult(
        status=RunStatus.SUCCESS,
        metadata=RunMetadata(
            run_id="run-id",
            scenario_name="smoke",
            started_at_utc="2026-03-24T14:32:10Z",
            finished_at_utc=None,
            hostname=None,
            python_version="3.13.0",
            platform="Linux-test",
            git_commit=None,
            cwd="/repo",
            command=[],
            notes={},
        ),
        timing=RunTimingSummary(wall_time_seconds=0.0),
        artifacts=RunArtifacts(run_json_path="run.json"),
    )

    serialized = run_result_to_dict(run_result)

    assert list(serialized) == [
        "schema_version",
        "status",
        "metadata",
        "timing",
        "artifacts",
        "error_message",
    ]
    assert json.loads(json.dumps(serialized))["schema_version"] == "1"
