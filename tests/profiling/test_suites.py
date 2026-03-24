"""Tests for profiling suite registration, execution, and CLI support."""

from pathlib import Path

import pytest

from anemone.profiling import cli, scenarios, suites
from anemone.profiling.artifacts import RunStatus
from anemone.profiling.scenarios import ProfilingScenario
from anemone.profiling.suite_artifacts import load_suite_run_result
from anemone.profiling.suite_runner import run_suite
from anemone.profiling.suites import ProfilingSuite, get_suite, list_suites


def test_baseline_suite_is_registered() -> None:
    """The baseline suite should include the representative synthetic scenarios."""
    baseline = get_suite("baseline")

    assert baseline.scenario_names == (
        "cheap_eval",
        "expensive_eval",
        "wide_tree",
        "deep_tree",
        "reuse_heavy",
    )


def test_list_suites_includes_registered_suites() -> None:
    """The suite registry should expose the standard suite names."""
    suite_names = {suite.name for suite in list_suites()}

    assert {"baseline", "quick"}.issubset(suite_names)


def test_run_suite_writes_baseline_suite_artifact(tmp_path: Path) -> None:
    """Running the baseline suite should produce stable suite-level artifacts."""
    result = run_suite("baseline", tmp_path, repetitions=2)

    suite_json_path = tmp_path / result.run_id / "suite.json"
    assert suite_json_path.exists()

    loaded = load_suite_run_result(suite_json_path)
    assert loaded.suite_name == "baseline"
    assert loaded.requested_repetitions == 2
    assert loaded.error_message is None
    assert len(loaded.scenario_runs) == 10
    assert len(loaded.scenario_aggregates) == 5
    assert all(
        aggregate.completed_repetitions == 2 for aggregate in loaded.scenario_aggregates
    )
    assert all(
        aggregate.failed_repetitions == 0 for aggregate in loaded.scenario_aggregates
    )
    assert all(
        len(aggregate.run_paths) == 2 for aggregate in loaded.scenario_aggregates
    )


def test_run_suite_records_component_summary_paths_when_requested(
    tmp_path: Path,
) -> None:
    """Suite runs should expose component summaries for supported scenarios."""
    result = run_suite(
        "quick",
        tmp_path,
        repetitions=1,
        component_summary=True,
    )

    suite_json_path = tmp_path / result.run_id / "suite.json"
    loaded = load_suite_run_result(suite_json_path)

    assert loaded.component_summary is True
    assert len(loaded.scenario_runs) == 2
    for scenario_run in loaded.scenario_runs:
        assert scenario_run.status is RunStatus.SUCCESS
        assert scenario_run.component_summary_json_path is not None


def test_run_suite_records_failed_repetitions_without_aborting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing repetition should still be represented in the suite artifact."""

    def _raise_failure() -> None:
        raise ValueError("boom")

    failing_scenario = ProfilingScenario(
        name="always_fail_suite",
        description="Scenario that always raises.",
        runner=_raise_failure,
    )
    failing_suite = ProfilingSuite(
        name="failing_suite",
        description="Suite used to verify repetition failure recording.",
        scenario_names=("always_fail_suite",),
    )
    monkeypatch.setitem(scenarios._SCENARIOS, failing_scenario.name, failing_scenario)
    monkeypatch.setitem(suites._SUITES, failing_suite.name, failing_suite)

    result = run_suite("failing_suite", tmp_path, repetitions=2)

    suite_json_path = tmp_path / result.run_id / "suite.json"
    loaded = load_suite_run_result(suite_json_path)
    assert loaded.error_message is None
    assert len(loaded.scenario_runs) == 2
    assert all(run.status is RunStatus.FAILED for run in loaded.scenario_runs)
    assert loaded.scenario_aggregates[0].failed_repetitions == 2
    assert loaded.scenario_aggregates[0].successful_repetitions == 0


def test_cli_list_suites_prints_registered_suites(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should list available profiling suites."""
    exit_code = cli.main(["list-suites"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "baseline:" in captured.out


def test_cli_run_suite_prints_suite_json_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should print the resulting suite artifact path."""
    exit_code = cli.main(
        [
            "run-suite",
            "--suite",
            "quick",
            "--output-dir",
            str(tmp_path),
            "--repetitions",
            "1",
            "--component-summary",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    suite_json_path = captured.out.strip()
    loaded = load_suite_run_result(suite_json_path)
    assert loaded.suite_name == "quick"
