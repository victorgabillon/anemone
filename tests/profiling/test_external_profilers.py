"""Tests for external profiler adapter helpers."""

from importlib.util import find_spec
from pathlib import Path

import pytest

from anemone.profiling.external_profilers import run_with_external_profiler


def test_run_with_external_profiler_cprofile_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    """The cProfile adapter should write stable artifact files."""

    def _target() -> None:
        total = 0
        for index in range(20):
            total += index
        assert total >= 0

    result = run_with_external_profiler("cprofile", _target, tmp_path)

    assert result.captured_error is None
    assert result.execution_wall_time_seconds >= 0.0
    assert set(result.extra_paths) == {"cprofile_pstats", "cprofile_top_txt"}
    assert (tmp_path / "cprofile.pstats").exists()
    assert (tmp_path / "cprofile_top.txt").exists()


def test_run_with_external_profiler_pyinstrument_handles_missing_dependency(
    tmp_path: Path,
) -> None:
    """An explicit pyinstrument request should fail cleanly when unavailable."""
    if find_spec("pyinstrument") is not None:
        pytest.skip("pyinstrument is installed in this environment")

    result = run_with_external_profiler("pyinstrument", lambda: None, tmp_path)

    assert result.execution_wall_time_seconds == 0.0
    assert result.extra_paths == {}
    assert isinstance(result.captured_error, RuntimeError)
    assert str(result.captured_error) == (
        "pyinstrument profiler requested but pyinstrument is not installed"
    )
