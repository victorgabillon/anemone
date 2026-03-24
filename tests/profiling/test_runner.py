"""Tests for profiling runner and CLI behavior."""

from importlib.util import find_spec
from pathlib import Path

import pytest

from anemone.profiling import cli, scenarios
from anemone.profiling.artifacts import RunStatus
from anemone.profiling.component_summary import load_component_summary
from anemone.profiling.runner import run_scenario
from anemone.profiling.scenarios import ProfilingScenario
from anemone.profiling.storage import load_run_result


def test_run_scenario_writes_success_artifact_for_smoke_scenario(
    tmp_path: Path,
) -> None:
    """The smoke scenario should produce a successful run artifact."""
    result = run_scenario("smoke", tmp_path)

    assert result.status is RunStatus.SUCCESS
    assert result.artifacts.run_json_path is not None

    run_json_path = tmp_path / result.metadata.run_id / "run.json"
    assert run_json_path.exists()
    loaded = load_run_result(run_json_path)
    assert loaded.status is RunStatus.SUCCESS
    assert loaded.metadata.scenario_name == "smoke"
    assert loaded.metadata.command == []
    assert loaded.metadata.notes["scenario_description"]
    assert loaded.metadata.notes["wall_time_definition"] == (
        "scenario execution wall time excluding profiler artifact writing"
    )
    assert loaded.timing.wall_time_seconds >= 0.0


def test_run_scenario_writes_component_summary_for_smoke_scenario(
    tmp_path: Path,
) -> None:
    """The smoke scenario should produce a component summary when requested."""
    result = run_scenario("smoke", tmp_path, component_summary=True)

    assert result.status is RunStatus.SUCCESS
    component_summary_path = (
        tmp_path / result.metadata.run_id / "component_summary.json"
    )
    assert component_summary_path.exists()
    assert result.artifacts.extra_paths["component_summary_json"] == str(
        component_summary_path
    )

    summary = load_component_summary(component_summary_path)
    assert summary.total_run_wall_time_seconds >= 0.0
    assert summary.total_profiled_component_wall_time_seconds >= 0.0
    assert summary.residual_framework_wall_time_seconds is not None
    assert summary.evaluator is not None
    assert summary.dynamics_step is not None
    assert summary.dynamics_legal_actions is not None


def test_run_scenario_writes_cprofile_artifacts_for_smoke_scenario(
    tmp_path: Path,
) -> None:
    """The smoke scenario should produce cProfile artifacts when requested."""
    result = run_scenario("smoke", tmp_path, profiler="cprofile")

    assert result.status is RunStatus.SUCCESS
    run_dir = tmp_path / result.metadata.run_id
    assert (run_dir / "cprofile.pstats").exists()
    assert (run_dir / "cprofile_top.txt").exists()
    assert result.artifacts.extra_paths["cprofile_pstats"] == str(
        run_dir / "cprofile.pstats"
    )
    assert result.artifacts.extra_paths["cprofile_top_txt"] == str(
        run_dir / "cprofile_top.txt"
    )


def test_run_scenario_persists_failure_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing scenario should still write a failed run artifact."""

    def _raise_failure() -> None:
        raise ValueError("boom")

    failing = ProfilingScenario(
        name="always_fail",
        description="Scenario that raises immediately.",
        runner=_raise_failure,
    )
    monkeypatch.setitem(scenarios._SCENARIOS, failing.name, failing)

    result = run_scenario("always_fail", tmp_path)

    assert result.status is RunStatus.FAILED
    assert result.error_message == "Scenario execution failed: ValueError: boom"

    run_json_path = tmp_path / result.metadata.run_id / "run.json"
    loaded = load_run_result(run_json_path)
    assert loaded.status is RunStatus.FAILED
    assert loaded.error_message == "Scenario execution failed: ValueError: boom"


def test_run_scenario_marks_run_failed_when_pyinstrument_is_unavailable(
    tmp_path: Path,
) -> None:
    """Explicit pyinstrument requests should still produce a failed run artifact."""
    if find_spec("pyinstrument") is not None:
        pytest.skip("pyinstrument is installed in this environment")

    result = run_scenario("smoke", tmp_path, profiler="pyinstrument")

    assert result.status is RunStatus.FAILED
    assert result.artifacts.run_json_path is not None
    assert (
        result.error_message
        == "Scenario execution failed: RuntimeError: pyinstrument profiler requested but pyinstrument is not installed"
    )
    run_json_path = tmp_path / result.metadata.run_id / "run.json"
    assert run_json_path.exists()


def test_cli_list_scenarios_prints_registered_scenarios(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should list available profiling scenarios."""
    exit_code = cli.main(["list-scenarios"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "smoke:" in captured.out


def test_cli_run_prints_run_json_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI run command should print the resulting run artifact path."""
    exit_code = cli.main(["run", "--scenario", "smoke", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip().endswith("/run.json")


def test_cli_run_with_cprofile_and_component_summary_writes_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should support cProfile and wrapper-based component summaries."""
    exit_code = cli.main(
        [
            "run",
            "--scenario",
            "smoke",
            "--output-dir",
            str(tmp_path),
            "--profiler",
            "cprofile",
            "--component-summary",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    run_json_path = captured.out.strip()
    loaded = load_run_result(run_json_path)
    run_dir = tmp_path / loaded.metadata.run_id
    assert (run_dir / "run.json").exists()
    assert (run_dir / "component_summary.json").exists()
    assert (run_dir / "cprofile.pstats").exists()
    assert loaded.artifacts.extra_paths["component_summary_json"] == str(
        run_dir / "component_summary.json"
    )
