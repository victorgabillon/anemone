"""Structured component-level timing summaries for profiling runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

COMPONENT_SUMMARY_FILENAME = "component_summary.json"


def _copy_str_dict(raw: object) -> dict[str, str]:
    """Return a plain string dictionary from loaded JSON-like input."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected a mapping, got {type(raw)}")
    return {str(key): str(value) for key, value in raw.items()}


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
            call_count=int(data["call_count"]),
            total_wall_time_seconds=float(data["total_wall_time_seconds"]),
            max_wall_time_seconds=float(data["max_wall_time_seconds"]),
            min_wall_time_seconds=float(min_raw) if min_raw is not None else None,
            mean_wall_time_seconds=float(data["mean_wall_time_seconds"]),
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
    notes: dict[str, str] = field(default_factory=dict)

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
            total_run_wall_time_seconds=float(data["total_run_wall_time_seconds"]),
            total_profiled_component_wall_time_seconds=float(
                data["total_profiled_component_wall_time_seconds"]
            ),
            residual_framework_wall_time_seconds=(
                float(residual_raw) if residual_raw is not None else None
            ),
            evaluator=(
                TimedCallStats.from_dict(evaluator_raw)
                if isinstance(evaluator_raw, Mapping)
                else None
            ),
            dynamics_step=(
                TimedCallStats.from_dict(step_raw)
                if isinstance(step_raw, Mapping)
                else None
            ),
            dynamics_legal_actions=(
                TimedCallStats.from_dict(legal_raw)
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
        raise TypeError(
            f"Expected component summary JSON object, got {type(loaded)}"
        )
    return ComponentSummary.from_dict(loaded)
