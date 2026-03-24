"""Structured component-level timing summaries for profiling runs."""
# pylint: disable=duplicate-code

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

COMPONENT_SUMMARY_FILENAME = "component_summary.json"


def _copy_str_dict(raw: object) -> dict[str, str]:
    """Return a plain string dictionary from loaded JSON-like input."""
    if raw is None:
        return {}
    mapping = _require_str_object_mapping(raw)
    return {str(key): str(value) for key, value in mapping.items()}


def _new_str_dict() -> dict[str, str]:
    return {}


def _expected_mapping_error(raw: object) -> TypeError:
    return TypeError(f"Expected a mapping, got {type(raw)}")


def _summary_object_error(raw: object) -> TypeError:
    return TypeError(f"Expected component summary JSON object, got {type(raw)}")


def _require_str_object_mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise _expected_mapping_error(raw)
    return cast("Mapping[str, object]", raw)


def _int_field(data: Mapping[str, object], field_name: str) -> int:
    raw_value = data[field_name]
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float | str):
        return int(raw_value)
    raise _numeric_field_error(field_name, raw_value, expected_type="int")


def _float_field(data: Mapping[str, object], field_name: str) -> float:
    raw_value = data[field_name]
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
class TimedCallStats:
    """Frozen summary of repeated timed calls."""

    call_count: int
    total_wall_time_seconds: float
    max_wall_time_seconds: float
    min_wall_time_seconds: float | None
    mean_wall_time_seconds: float

    def to_dict(self) -> dict[str, int | float | None]:
        """Serialize the timed-call statistics to JSON-friendly data."""
        return {
            "call_count": self.call_count,
            "total_wall_time_seconds": self.total_wall_time_seconds,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "min_wall_time_seconds": self.min_wall_time_seconds,
            "mean_wall_time_seconds": self.mean_wall_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TimedCallStats:
        """Deserialize the timed-call statistics from JSON-friendly data."""
        min_raw = data.get("min_wall_time_seconds")
        return cls(
            call_count=_int_field(data, "call_count"),
            total_wall_time_seconds=_float_field(data, "total_wall_time_seconds"),
            max_wall_time_seconds=_float_field(data, "max_wall_time_seconds"),
            min_wall_time_seconds=(
                float(min_raw)
                if isinstance(min_raw, str | int | float | bool)
                else None
                if min_raw is None
                else _raise_min_wall_time_error(min_raw)
            ),
            mean_wall_time_seconds=_float_field(data, "mean_wall_time_seconds"),
        )


@dataclass(slots=True)
class ComponentSummary:
    """Stable artifact describing wrapper-based component timings."""

    total_run_wall_time_seconds: float
    total_profiled_component_wall_time_seconds: float
    residual_framework_wall_time_seconds: float | None
    evaluator: TimedCallStats | None = None
    dynamics_step: TimedCallStats | None = None
    dynamics_legal_actions: TimedCallStats | None = None
    notes: dict[str, str] = field(default_factory=_new_str_dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the component summary to JSON-friendly data."""
        return {
            "total_run_wall_time_seconds": self.total_run_wall_time_seconds,
            "total_profiled_component_wall_time_seconds": (
                self.total_profiled_component_wall_time_seconds
            ),
            "residual_framework_wall_time_seconds": (
                self.residual_framework_wall_time_seconds
            ),
            "evaluator": None if self.evaluator is None else self.evaluator.to_dict(),
            "dynamics_step": (
                None if self.dynamics_step is None else self.dynamics_step.to_dict()
            ),
            "dynamics_legal_actions": (
                None
                if self.dynamics_legal_actions is None
                else self.dynamics_legal_actions.to_dict()
            ),
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ComponentSummary:
        """Deserialize the component summary from JSON-friendly data."""
        evaluator_raw = data.get("evaluator")
        step_raw = data.get("dynamics_step")
        legal_raw = data.get("dynamics_legal_actions")
        residual_raw = data.get("residual_framework_wall_time_seconds")
        return cls(
            total_run_wall_time_seconds=_float_field(
                data, "total_run_wall_time_seconds"
            ),
            total_profiled_component_wall_time_seconds=_float_field(
                data, "total_profiled_component_wall_time_seconds"
            ),
            residual_framework_wall_time_seconds=(
                float(residual_raw)
                if isinstance(residual_raw, str | int | float | bool)
                else None
                if residual_raw is None
                else _raise_residual_error(residual_raw)
            ),
            evaluator=(
                TimedCallStats.from_dict(cast("Mapping[str, object]", evaluator_raw))
                if isinstance(evaluator_raw, Mapping)
                else None
            ),
            dynamics_step=(
                TimedCallStats.from_dict(cast("Mapping[str, object]", step_raw))
                if isinstance(step_raw, Mapping)
                else None
            ),
            dynamics_legal_actions=(
                TimedCallStats.from_dict(cast("Mapping[str, object]", legal_raw))
                if isinstance(legal_raw, Mapping)
                else None
            ),
            notes=_copy_str_dict(data.get("notes")),
        )


def save_component_summary(summary: ComponentSummary, path: Path) -> None:
    """Persist a component summary as pretty UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_component_summary(path: Path) -> ComponentSummary:
    """Load a component summary artifact from disk."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise _summary_object_error(loaded)
    return ComponentSummary.from_dict(cast("dict[str, object]", loaded))


def _raise_min_wall_time_error(raw: object) -> float:
    raise _numeric_field_error(
        "min_wall_time_seconds",
        raw,
        expected_type="float",
    )


def _raise_residual_error(raw: object) -> float | None:
    raise _numeric_field_error(
        "residual_framework_wall_time_seconds",
        raw,
        expected_type="float",
    )
