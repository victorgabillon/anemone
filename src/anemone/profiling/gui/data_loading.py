"""Artifact loading helpers for the local profiling dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from anemone.profiling.component_summary import (
    COMPONENT_SUMMARY_FILENAME,
    load_component_summary,
)
from anemone.profiling.storage import RUN_JSON_FILENAME, load_run_result
from anemone.profiling.suite_artifacts import (
    SUITE_JSON_FILENAME,
    load_suite_run_result,
)

if TYPE_CHECKING:
    from anemone.profiling.artifacts import RunResult
    from anemone.profiling.component_summary import ComponentSummary
    from anemone.profiling.suite_artifacts import (
        ScenarioRepetitionSummary,
        SuiteRunResult,
    )

_GUI_RUN_JSON_SOURCE_NOTE = "_gui_run_json_source_path"
_GUI_SUITE_JSON_SOURCE_NOTE = "_gui_suite_json_source_path"


def load_run(run_dir: Path) -> RunResult:
    """Load one profiling run from its run directory or artifact path."""
    run_json_path = _resolve_run_json_path(run_dir).resolve()
    run = load_run_result(run_json_path)
    _record_run_source_path(run, run_json_path)
    return run


def load_suite(suite_dir: Path) -> SuiteRunResult:
    """Load one profiling suite from its suite directory or artifact path."""
    suite_json_path = _resolve_suite_json_path(suite_dir).resolve()
    suite = load_suite_run_result(suite_json_path)
    _record_suite_source_path(suite, suite_json_path)
    return suite


def discover_runs(base_dir: Path) -> list[RunResult]:
    """Discover all readable ``run.json`` artifacts below one base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    runs: list[RunResult] = []
    for run_json_path in base_path.rglob(RUN_JSON_FILENAME):
        try:
            run = load_run_result(run_json_path)
            _record_run_source_path(run, run_json_path.resolve())
            runs.append(run)
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            continue
    return sorted(
        runs,
        key=lambda run: (run.metadata.started_at_utc, run.metadata.run_id),
        reverse=True,
    )


def discover_suites(base_dir: Path) -> list[SuiteRunResult]:
    """Discover all readable ``suite.json`` artifacts below one base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    suites: list[SuiteRunResult] = []
    for suite_json_path in base_path.rglob(SUITE_JSON_FILENAME):
        try:
            suite = load_suite_run_result(suite_json_path)
            _record_suite_source_path(suite, suite_json_path.resolve())
            suites.append(suite)
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            continue
    return sorted(
        suites,
        key=lambda suite: (suite.started_at_utc, suite.run_id),
        reverse=True,
    )


def extract_component_summary(run_dir: Path) -> ComponentSummary | None:
    """Load the component summary for one run when it exists and is readable."""
    run = load_run(run_dir)
    candidate = _resolve_run_artifact_path(
        run,
        run.artifacts.extra_paths.get("component_summary_json"),
        default_name=COMPONENT_SUMMARY_FILENAME,
    )
    if candidate is None or not candidate.exists():
        return None
    try:
        return load_component_summary(candidate)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run_dir_from_result(run: RunResult) -> Path | None:
    """Return the run directory for one loaded run artifact when known."""
    source_path = _run_source_path(run)
    if source_path is not None:
        return source_path.parent

    run_json_path = _resolve_run_artifact_path(run, run.artifacts.run_json_path)
    return None if run_json_path is None else run_json_path.parent


def suite_dir_from_result(base_dir: Path, suite: SuiteRunResult) -> Path:
    """Return the suite directory under the chosen base directory."""
    source_path = _suite_source_path(suite)
    return (
        source_path.parent if source_path is not None else Path(base_dir) / suite.run_id
    )


def read_text_artifact(
    path: str | Path | None,
    *,
    run: RunResult | None = None,
    anchor_path: Path | None = None,
) -> str | None:
    """Read one optional text artifact, returning ``None`` when missing."""
    candidate = (
        _resolve_run_artifact_path(run, path)
        if run is not None
        else _resolve_artifact_path(path, source_artifact_path=anchor_path)
    )
    if candidate is None or not candidate.exists():
        return None
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError:
        return None


def resolve_run_artifact_path(
    run: RunResult,
    path: str | Path | None,
    *,
    default_name: str | None = None,
) -> Path | None:
    """Resolve one run-scoped artifact path for reuse in view helpers."""
    return _resolve_run_artifact_path(run, path, default_name=default_name)


def resolve_suite_artifact_path(
    suite: SuiteRunResult,
    path: str | Path | None,
) -> Path | None:
    """Resolve one suite-scoped artifact path for linked drill-downs."""
    return _resolve_artifact_path(
        path,
        source_artifact_path=_suite_source_path(suite),
    )


def load_suite_repetition_run(
    suite: SuiteRunResult,
    repetition: ScenarioRepetitionSummary,
) -> RunResult | None:
    """Load the run linked from one suite repetition when it can be resolved."""
    run_json_path = resolve_suite_artifact_path(suite, repetition.run_json_path)
    if run_json_path is None or not run_json_path.exists():
        return None
    try:
        run = load_run_result(run_json_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    _record_run_source_path(run, run_json_path.resolve())
    return run


def _resolve_run_json_path(run_dir: Path) -> Path:
    candidate = Path(run_dir)
    return (
        candidate
        if candidate.name == RUN_JSON_FILENAME
        else candidate / RUN_JSON_FILENAME
    )


def _resolve_suite_json_path(suite_dir: Path) -> Path:
    candidate = Path(suite_dir)
    return (
        candidate
        if candidate.name == SUITE_JSON_FILENAME
        else candidate / SUITE_JSON_FILENAME
    )


def _resolve_run_artifact_path(
    run: RunResult,
    path: str | Path | None,
    *,
    default_name: str | None = None,
) -> Path | None:
    stored_path = default_name if path is None else path
    return _resolve_artifact_path(
        stored_path,
        source_artifact_path=_run_source_path(run),
        fallback_cwd=Path(run.metadata.cwd),
    )


def _resolve_artifact_path(
    path: str | Path | None,
    *,
    source_artifact_path: Path | None = None,
    fallback_cwd: Path | None = None,
) -> Path | None:
    if path is None:
        return None

    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        return raw_path

    for candidate in _candidate_paths(
        raw_path,
        source_artifact_path=source_artifact_path,
        fallback_cwd=fallback_cwd,
    ):
        if candidate.exists():
            return candidate.resolve()

    if fallback_cwd is not None:
        return (fallback_cwd / raw_path).resolve(strict=False)
    if source_artifact_path is not None:
        return (source_artifact_path.parent / raw_path.name).resolve(strict=False)
    return raw_path.resolve(strict=False)


def _candidate_paths(
    raw_path: Path,
    *,
    source_artifact_path: Path | None,
    fallback_cwd: Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    if source_artifact_path is not None:
        for anchor in [source_artifact_path.parent, *source_artifact_path.parents]:
            candidate = (anchor / raw_path).resolve(strict=False)
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
        sibling_candidate = (source_artifact_path.parent / raw_path.name).resolve(
            strict=False
        )
        if sibling_candidate not in seen:
            candidates.append(sibling_candidate)
            seen.add(sibling_candidate)

    if fallback_cwd is not None:
        fallback_candidate = (fallback_cwd / raw_path).resolve(strict=False)
        if fallback_candidate not in seen:
            candidates.append(fallback_candidate)
            seen.add(fallback_candidate)

    return candidates


def _record_run_source_path(run: RunResult, run_json_path: Path) -> None:
    run.metadata.notes[_GUI_RUN_JSON_SOURCE_NOTE] = str(run_json_path.resolve())


def _record_suite_source_path(suite: SuiteRunResult, suite_json_path: Path) -> None:
    suite.notes[_GUI_SUITE_JSON_SOURCE_NOTE] = str(suite_json_path.resolve())


def _run_source_path(run: RunResult) -> Path | None:
    source_path = run.metadata.notes.get(_GUI_RUN_JSON_SOURCE_NOTE)
    return None if source_path is None else Path(source_path)


def _suite_source_path(suite: SuiteRunResult) -> Path | None:
    source_path = suite.notes.get(_GUI_SUITE_JSON_SOURCE_NOTE)
    return None if source_path is None else Path(source_path)
