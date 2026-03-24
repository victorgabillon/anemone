"""Tests for deterministic profiling scenario registration and execution."""

from pathlib import Path

import pytest

from anemone.profiling.runner import run_scenario
from anemone.profiling.scenarios import get_scenario, list_scenarios
from anemone.profiling.storage import load_run_result


def test_list_scenarios_includes_deterministic_benchmark_scenarios() -> None:
    """The scenario registry should expose the synthetic profiling scenarios."""
    scenario_names = {scenario.name for scenario in list_scenarios()}

    assert {
        "smoke",
        "cheap_eval",
        "expensive_eval",
        "wide_tree",
        "deep_tree",
        "reuse_heavy",
    }.issubset(scenario_names)


@pytest.mark.parametrize(
    "scenario_name",
    [
        "cheap_eval",
        "expensive_eval",
        "wide_tree",
        "deep_tree",
        "reuse_heavy",
    ],
)
def test_synthetic_scenarios_run_successfully(
    tmp_path: Path,
    scenario_name: str,
) -> None:
    """Each deterministic synthetic scenario should run successfully once."""
    scenario = get_scenario(scenario_name)

    result = run_scenario(scenario.name, tmp_path)

    run_json_path = tmp_path / result.metadata.run_id / "run.json"
    assert result.status.value == "success"
    assert run_json_path.exists()
    loaded = load_run_result(run_json_path)
    assert loaded.metadata.scenario_name == scenario_name
    assert loaded.metadata.notes["scenario_description"] == scenario.description
    assert loaded.timing.wall_time_seconds >= 0.0


def test_run_scenario_uses_scenario_label_only_for_run_directory_naming(
    tmp_path: Path,
) -> None:
    """Scenario labels should affect directory naming without changing metadata."""
    result = run_scenario(
        "cheap_eval",
        tmp_path,
        scenario_label="cheap_eval_rep1",
    )

    assert result.metadata.scenario_name == "cheap_eval"
    assert result.metadata.notes["scenario_label"] == "cheap_eval_rep1"
    assert result.metadata.run_id.endswith("_cheap_eval_rep1")
