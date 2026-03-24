"""Filesystem helpers for profiling run directories and JSON artifacts."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import RunResult, run_result_to_dict

RUN_JSON_FILENAME = "run.json"

_INVALID_SCENARIO_CHARS = re.compile(r"[^a-z0-9._-]+")
_MULTIPLE_UNDERSCORES = re.compile(r"_+")


def sanitize_scenario_name(scenario_name: str) -> str:
    """Return a filesystem-safe scenario name for use inside run identifiers."""
    lowered = scenario_name.strip().lower()
    spaced = re.sub(r"\s+", "_", lowered)
    safe = _INVALID_SCENARIO_CHARS.sub("_", spaced)
    collapsed = _MULTIPLE_UNDERSCORES.sub("_", safe)
    cleaned = collapsed.strip("._-")
    return cleaned or "scenario"


def make_run_id(
    scenario_name: str,
    *,
    timestamp: datetime | None = None,
) -> str:
    """Create a readable, sortable run identifier."""
    run_timestamp = datetime.now(tz=UTC) if timestamp is None else timestamp.astimezone(UTC)
    prefix = run_timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    return f"{prefix}_{sanitize_scenario_name(scenario_name)}"


def default_runs_base_dir() -> Path:
    """Return the default base directory for profiling runs."""
    return Path("profiling_runs")


def make_run_dir(base_dir: Path, run_id: str) -> Path:
    """Create and return the directory that will hold one run's artifacts.

    If the preferred run identifier already exists, append ``_2``, ``_3``, and
    so on until a free directory name is found.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    suffix = 1
    while True:
        candidate_run_id = run_id if suffix == 1 else f"{run_id}_{suffix}"
        candidate = base_path / candidate_run_id
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def run_result_path(run_dir: Path) -> Path:
    """Return the canonical JSON artifact path for one run directory."""
    return Path(run_dir) / RUN_JSON_FILENAME


def save_run_result(run_result: RunResult, path: Path) -> None:
    """Persist a run result as pretty UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(run_result_to_dict(run_result), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_run_result(path: Path) -> RunResult:
    """Load a run result from disk."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise TypeError(f"Expected run artifact JSON object, got {type(loaded)}")

    return RunResult.from_dict(loaded)
