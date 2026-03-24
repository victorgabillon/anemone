"""Tests for profiling runner and CLI behavior."""

from anemone.profiling import cli
from anemone.profiling.artifacts import RunStatus
from anemone.profiling.runner import run_scenario
from anemone.profiling import scenarios
from anemone.profiling.scenarios import ProfilingScenario
from anemone.profiling.storage import load_run_result


def test_run_scenario_writes_success_artifact_for_smoke_scenario(tmp_path) -> None:
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
    assert loaded.timing.wall_time_seconds >= 0.0


def test_run_scenario_persists_failure_artifact(tmp_path, monkeypatch) -> None:
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


def test_cli_list_scenarios_prints_registered_scenarios(capsys) -> None:
    """The CLI should list available profiling scenarios."""
    exit_code = cli.main(["list-scenarios"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "smoke:" in captured.out


def test_cli_run_prints_run_json_path(tmp_path, capsys) -> None:
    """The CLI run command should print the resulting run artifact path."""
    exit_code = cli.main(
        ["run", "--scenario", "smoke", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip().endswith("/run.json")
