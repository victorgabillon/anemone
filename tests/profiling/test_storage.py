"""Tests for profiling run-id and directory helpers."""

from datetime import UTC, datetime
from pathlib import Path

from anemone.profiling.storage import make_run_dir, make_run_id


def test_make_run_id_sanitizes_scenario_name() -> None:
    """Run identifiers should remain readable and filesystem-safe."""
    run_id = make_run_id(
        " Smoke Scenario!? / v1 ",
        timestamp=datetime(2026, 3, 24, 14, 32, 10, tzinfo=UTC),
    )

    assert run_id == "2026-03-24T14-32-10_smoke_scenario_v1"


def test_make_run_dir_creates_expected_directory(tmp_path: Path) -> None:
    """Creating a run directory should materialize the expected path."""
    run_dir = make_run_dir(tmp_path / "profiling_runs", "2026-03-24T14-32-10_smoke")

    assert run_dir.exists()
    assert run_dir.is_dir()


def test_make_run_dir_appends_suffix_on_collision(tmp_path: Path) -> None:
    """Repeated run ids should allocate deterministic suffixed directories."""
    base_dir = tmp_path / "profiling_runs"

    first = make_run_dir(base_dir, "2026-03-24T14-32-10_smoke")
    second = make_run_dir(base_dir, "2026-03-24T14-32-10_smoke")

    assert first.name == "2026-03-24T14-32-10_smoke"
    assert second.name == "2026-03-24T14-32-10_smoke_2"
