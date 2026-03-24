"""Small adapters for optional external profiler execution."""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Literal

ProfilerName = Literal["none", "cprofile", "pyinstrument"]
SUPPORTED_EXTERNAL_PROFILERS: tuple[ProfilerName, ...] = (
    "none",
    "cprofile",
    "pyinstrument",
)


@dataclass(slots=True)
class ExternalProfilerRunResult:
    """Stable result for one externally-profiled execution."""

    execution_wall_time_seconds: float
    extra_paths: dict[str, str] = field(default_factory=dict)
    captured_error: Exception | None = None


def run_with_external_profiler(
    profiler_name: ProfilerName,
    target: Callable[[], None],
    run_dir: Path,
) -> ExternalProfilerRunResult:
    """Execute one target callable under the requested external profiler."""
    if profiler_name == "none":
        return _run_without_external_profiler(target)
    if profiler_name == "cprofile":
        return _run_with_cprofile(target, Path(run_dir))
    if profiler_name == "pyinstrument":
        return _run_with_pyinstrument(target, Path(run_dir))

    supported = ", ".join(SUPPORTED_EXTERNAL_PROFILERS)
    raise ValueError(
        f"Unsupported profiler {profiler_name!r}. Supported profilers: {supported}"
    )


def _run_without_external_profiler(
    target: Callable[[], None],
) -> ExternalProfilerRunResult:
    return _run_profiled_target(target)


def _run_profiled_target(target: Callable[[], None]) -> ExternalProfilerRunResult:
    started_at = time.perf_counter()
    captured_error: Exception | None = None
    try:
        target()
    except Exception as exc:  # pragma: no cover - surfaced via runner tests
        captured_error = exc
    execution_wall_time_seconds = time.perf_counter() - started_at
    return ExternalProfilerRunResult(
        execution_wall_time_seconds=execution_wall_time_seconds,
        captured_error=captured_error,
    )


def _run_with_cprofile(
    target: Callable[[], None],
    run_dir: Path,
) -> ExternalProfilerRunResult:
    profiler = cProfile.Profile()
    profiler.enable()
    execution_result: ExternalProfilerRunResult
    try:
        execution_result = _run_profiled_target(target)
    finally:
        profiler.disable()

        pstats_path = run_dir / "cprofile.pstats"
        top_txt_path = run_dir / "cprofile_top.txt"
        profiler.dump_stats(str(pstats_path))

        output_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=output_stream)
        stats.sort_stats("cumulative")
        stats.print_stats(40)
        top_txt_path.write_text(output_stream.getvalue(), encoding="utf-8")

    execution_result.extra_paths.update(
        {
            "cprofile_pstats": str(pstats_path),
            "cprofile_top_txt": str(top_txt_path),
        }
    )
    return execution_result


def _run_with_pyinstrument(
    target: Callable[[], None],
    run_dir: Path,
) -> ExternalProfilerRunResult:
    try:
        pyinstrument_module = import_module("pyinstrument")
    except ImportError:
        return ExternalProfilerRunResult(
            execution_wall_time_seconds=0.0,
            captured_error=RuntimeError(
                "pyinstrument profiler requested but pyinstrument is not installed"
            ),
        )

    profiler = pyinstrument_module.Profiler()
    profiler.start()
    execution_result: ExternalProfilerRunResult
    try:
        execution_result = _run_profiled_target(target)
    finally:
        profiler.stop()
        output_text = profiler.output_text(unicode=True, color=False)
        output_path = run_dir / "pyinstrument.txt"
        output_path.write_text(output_text, encoding="utf-8")

    execution_result.extra_paths["pyinstrument_txt"] = str(output_path)
    return execution_result
