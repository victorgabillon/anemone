"""Stable JSON artifact models for profiling suites."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import cast

from .artifacts import RunStatus

SUITE_SCHEMA_VERSION = "1"
SUITE_JSON_FILENAME = "suite.json"


def _new_str_dict() -> dict[str, str]:
    return {}


def _new_str_list() -> list[str]:
    return []


def _new_repetition_list() -> list[ScenarioRepetitionSummary]:
    return []


def _new_aggregate_list() -> list[ScenarioAggregateSummary]:
    return []


def _expected_mapping_error(raw: object) -> TypeError:
    return TypeError(f"Expected a mapping, got {type(raw)}")


def _expected_list_error(raw: object) -> TypeError:
    return TypeError(f"Expected a list, got {type(raw)}")


def _suite_artifact_object_error(raw: object) -> TypeError:
    return TypeError(f"Expected suite artifact JSON object, got {type(raw)}")


def _require_str_object_mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise _expected_mapping_error(raw)
    return cast("Mapping[str, object]", raw)


def _copy_str_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise _expected_list_error(raw)
    list_raw = cast("list[object]", raw)
    return [str(item) for item in list_raw]


def _int_field(data: Mapping[str, object], field_name: str) -> int:
    raw_value = data[field_name]
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float | str):
        return int(raw_value)
    raise _numeric_field_error(field_name, raw_value, expected_type="int")


def _float_or_none_field(data: Mapping[str, object], field_name: str) -> float | None:
    raw_value = data.get(field_name)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return float(raw_value)
    if isinstance(raw_value, int | float):
        return float(raw_value)
    if isinstance(raw_value, str):
        return float(raw_value)
    raise _numeric_field_error(field_name, raw_value, expected_type="float")


def _numeric_field_error(
    field_name: str,
    raw_value: object,
    *,
    expected_type: str,
) -> TypeError:
    return TypeError(
        f"{field_name} must be {expected_type}-compatible, got {type(raw_value)}"
    )


@dataclass(slots=True)
class ScenarioRepetitionSummary:
    """Stable record for one suite scenario repetition."""

    scenario_name: str
    repetition_index: int
    status: RunStatus
    run_json_path: str | None
    wall_time_seconds: float | None
    component_summary_json_path: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize one repetition summary to JSON-friendly data."""
        return {
            "scenario_name": self.scenario_name,
            "repetition_index": self.repetition_index,
            "status": self.status.value,
            "run_json_path": self.run_json_path,
            "wall_time_seconds": self.wall_time_seconds,
            "component_summary_json_path": self.component_summary_json_path,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ScenarioRepetitionSummary:
        """Deserialize one repetition summary from JSON-friendly data."""
        run_json_path_raw = data.get("run_json_path")
        component_summary_raw = data.get("component_summary_json_path")
        error_raw = data.get("error_message")
        return cls(
            scenario_name=str(data["scenario_name"]),
            repetition_index=_int_field(data, "repetition_index"),
            status=RunStatus(str(data["status"])),
            run_json_path=str(run_json_path_raw)
            if run_json_path_raw is not None
            else None,
            wall_time_seconds=_float_or_none_field(data, "wall_time_seconds"),
            component_summary_json_path=(
                str(component_summary_raw)
                if component_summary_raw is not None
                else None
            ),
            error_message=str(error_raw) if error_raw is not None else None,
        )


@dataclass(slots=True)
class ScenarioAggregateSummary:
    """Stable aggregate wall-time summary for one scenario across repetitions."""

    scenario_name: str
    requested_repetitions: int
    completed_repetitions: int
    successful_repetitions: int
    failed_repetitions: int
    wall_time_min_seconds: float | None
    wall_time_max_seconds: float | None
    wall_time_mean_seconds: float | None
    wall_time_median_seconds: float | None
    run_paths: list[str] = field(default_factory=_new_str_list)

    def to_dict(self) -> dict[str, object]:
        """Serialize one aggregate summary to JSON-friendly data."""
        return {
            "scenario_name": self.scenario_name,
            "requested_repetitions": self.requested_repetitions,
            "completed_repetitions": self.completed_repetitions,
            "successful_repetitions": self.successful_repetitions,
            "failed_repetitions": self.failed_repetitions,
            "wall_time_min_seconds": self.wall_time_min_seconds,
            "wall_time_max_seconds": self.wall_time_max_seconds,
            "wall_time_mean_seconds": self.wall_time_mean_seconds,
            "wall_time_median_seconds": self.wall_time_median_seconds,
            "run_paths": list(self.run_paths),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ScenarioAggregateSummary:
        """Deserialize one aggregate summary from JSON-friendly data."""
        return cls(
            scenario_name=str(data["scenario_name"]),
            requested_repetitions=_int_field(data, "requested_repetitions"),
            completed_repetitions=_int_field(data, "completed_repetitions"),
            successful_repetitions=_int_field(data, "successful_repetitions"),
            failed_repetitions=_int_field(data, "failed_repetitions"),
            wall_time_min_seconds=_float_or_none_field(data, "wall_time_min_seconds"),
            wall_time_max_seconds=_float_or_none_field(data, "wall_time_max_seconds"),
            wall_time_mean_seconds=_float_or_none_field(data, "wall_time_mean_seconds"),
            wall_time_median_seconds=_float_or_none_field(
                data, "wall_time_median_seconds"
            ),
            run_paths=_copy_str_list(data.get("run_paths")),
        )

    @classmethod
    def from_repetitions(
        cls,
        scenario_name: str,
        repetitions: list[ScenarioRepetitionSummary],
        *,
        requested_repetitions: int,
    ) -> ScenarioAggregateSummary:
        """Build one aggregate summary from per-repetition records."""
        successful_wall_times = [
            repetition.wall_time_seconds
            for repetition in repetitions
            if repetition.status is RunStatus.SUCCESS
            and repetition.wall_time_seconds is not None
        ]
        run_paths = [
            repetition.run_json_path
            for repetition in repetitions
            if repetition.run_json_path is not None
        ]
        return cls(
            scenario_name=scenario_name,
            requested_repetitions=requested_repetitions,
            completed_repetitions=len(repetitions),
            successful_repetitions=sum(
                1
                for repetition in repetitions
                if repetition.status is RunStatus.SUCCESS
            ),
            failed_repetitions=sum(
                1 for repetition in repetitions if repetition.status is RunStatus.FAILED
            ),
            wall_time_min_seconds=(
                min(successful_wall_times) if successful_wall_times else None
            ),
            wall_time_max_seconds=(
                max(successful_wall_times) if successful_wall_times else None
            ),
            wall_time_mean_seconds=(
                mean(successful_wall_times) if successful_wall_times else None
            ),
            wall_time_median_seconds=(
                median(successful_wall_times) if successful_wall_times else None
            ),
            run_paths=run_paths,
        )


@dataclass(slots=True)
class SuiteRunResult:
    """Top-level persisted result for one profiling suite run."""

    suite_name: str
    description: str
    run_id: str
    started_at_utc: str
    finished_at_utc: str | None
    requested_repetitions: int
    profiler: str
    component_summary: bool
    scenario_aggregates: list[ScenarioAggregateSummary] = field(
        default_factory=_new_aggregate_list
    )
    scenario_runs: list[ScenarioRepetitionSummary] = field(
        default_factory=_new_repetition_list
    )
    error_message: str | None = None
    notes: dict[str, str] = field(default_factory=_new_str_dict)
    schema_version: str = SUITE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete suite result."""
        return {
            "schema_version": self.schema_version,
            "suite_name": self.suite_name,
            "description": self.description,
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "requested_repetitions": self.requested_repetitions,
            "profiler": self.profiler,
            "component_summary": self.component_summary,
            "scenario_aggregates": [
                aggregate.to_dict() for aggregate in self.scenario_aggregates
            ],
            "scenario_runs": [run.to_dict() for run in self.scenario_runs],
            "error_message": self.error_message,
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SuiteRunResult:
        """Deserialize the complete suite result."""
        scenario_aggregates_raw = data.get("scenario_aggregates", [])
        scenario_runs_raw = data.get("scenario_runs", [])
        if not isinstance(scenario_aggregates_raw, list):
            raise _expected_list_error(scenario_aggregates_raw)
        if not isinstance(scenario_runs_raw, list):
            raise _expected_list_error(scenario_runs_raw)

        aggregate_items = cast("list[object]", scenario_aggregates_raw)
        run_items = cast("list[object]", scenario_runs_raw)
        notes_raw = data.get("notes")
        return cls(
            schema_version=str(data.get("schema_version", SUITE_SCHEMA_VERSION)),
            suite_name=str(data["suite_name"]),
            description=str(data["description"]),
            run_id=str(data["run_id"]),
            started_at_utc=str(data["started_at_utc"]),
            finished_at_utc=(
                str(data["finished_at_utc"])
                if data.get("finished_at_utc") is not None
                else None
            ),
            requested_repetitions=_int_field(data, "requested_repetitions"),
            profiler=str(data["profiler"]),
            component_summary=bool(data["component_summary"]),
            scenario_aggregates=[
                ScenarioAggregateSummary.from_dict(_require_str_object_mapping(item))
                for item in aggregate_items
            ],
            scenario_runs=[
                ScenarioRepetitionSummary.from_dict(_require_str_object_mapping(item))
                for item in run_items
            ],
            error_message=(
                str(data["error_message"])
                if data.get("error_message") is not None
                else None
            ),
            notes=(
                {
                    str(key): str(value)
                    for key, value in _require_str_object_mapping(notes_raw).items()
                }
                if notes_raw is not None
                else {}
            ),
        )


def save_suite_run_result(result: SuiteRunResult, path: Path) -> None:
    """Persist a suite run result as pretty UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_suite_run_result(path: Path) -> SuiteRunResult:
    """Load a suite run result from disk."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise _suite_artifact_object_error(loaded)
    return SuiteRunResult.from_dict(cast("dict[str, object]", loaded))


def suite_result_path(run_dir: Path) -> Path:
    """Return the canonical suite artifact path for one suite run directory."""
    return Path(run_dir) / SUITE_JSON_FILENAME
