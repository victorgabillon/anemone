"""Tests for profiling GUI artifact-loading helpers."""

from pathlib import Path

import pytest

from anemone.profiling.gui.data_loading import (
    discover_runs,
    discover_suites,
    extract_component_summary,
    load_run,
    load_suite,
    read_text_artifact,
    run_dir_from_result,
)
from anemone.profiling.runner import run_scenario
from anemone.profiling.suite_runner import run_suite


def test_load_run_and_extract_component_summary(tmp_path: Path) -> None:
    """GUI loaders should read run artifacts and optional component summaries."""
    result = run_scenario("cheap_eval", tmp_path, component_summary=True)
    run_dir = tmp_path / result.metadata.run_id

    loaded_run = load_run(run_dir)
    summary = extract_component_summary(run_dir)

    assert loaded_run.metadata.run_id == result.metadata.run_id
    assert loaded_run.metadata.scenario_name == "cheap_eval"
    assert summary is not None
    assert summary.total_run_wall_time_seconds >= 0.0


def test_discover_runs_and_suites_find_written_artifacts(tmp_path: Path) -> None:
    """GUI discovery helpers should find standalone runs and suite artifacts."""
    standalone_result = run_scenario("cheap_eval", tmp_path)
    suite_result = run_suite("quick", tmp_path, repetitions=1)

    discovered_runs = discover_runs(tmp_path)
    discovered_suites = discover_suites(tmp_path)
    loaded_suite = load_suite(tmp_path / suite_result.run_id)

    discovered_run_ids = {run.metadata.run_id for run in discovered_runs}
    discovered_suite_ids = {suite.run_id for suite in discovered_suites}

    assert standalone_result.metadata.run_id in discovered_run_ids
    assert suite_result.run_id in discovered_suite_ids
    assert loaded_suite.run_id == suite_result.run_id
    assert len(discovered_runs) >= 3


def test_gui_resolves_relative_artifact_paths_after_cwd_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUI artifact helpers should not depend on the current working directory."""
    producer_cwd = tmp_path / "producer"
    consumer_cwd = tmp_path / "consumer"
    producer_cwd.mkdir()
    consumer_cwd.mkdir()

    monkeypatch.chdir(producer_cwd)
    result = run_scenario(
        "cheap_eval",
        Path("profiling_runs"),
        component_summary=True,
        profiler="cprofile",
    )
    base_dir = producer_cwd / "profiling_runs"

    monkeypatch.chdir(consumer_cwd)
    discovered_runs = discover_runs(base_dir)
    discovered_run = next(
        run for run in discovered_runs if run.metadata.run_id == result.metadata.run_id
    )

    run_dir = run_dir_from_result(discovered_run)
    assert run_dir == base_dir / result.metadata.run_id

    summary = extract_component_summary(run_dir)
    assert summary is not None

    cprofile_text = read_text_artifact(
        discovered_run.artifacts.extra_paths.get("cprofile_top_txt"),
        run=discovered_run,
    )
    assert cprofile_text is not None
