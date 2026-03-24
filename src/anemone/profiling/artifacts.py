"""Stable JSON artifact models for profiling runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1"


class RunStatus(str, Enum):
    """Terminal status for one profiling run."""

    SUCCESS = "success"
    FAILED = "failed"


def _copy_str_dict(raw: object) -> dict[str, str]:
    """Return a plain string dictionary from loaded JSON-like input."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected a mapping, got {type(raw)}")
    return {str(key): str(value) for key, value in raw.items()}


def _copy_str_list(raw: object) -> list[str]:
    """Return a plain string list from loaded JSON-like input."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError(f"Expected a list, got {type(raw)}")
    return [str(item) for item in raw]


@dataclass(slots=True)
class RunArtifacts:
    """References to files produced by one profiling run."""

    run_json_path: str | None = None
    extra_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the artifact references to JSON-friendly data."""
        return {
            "run_json_path": self.run_json_path,
            "extra_paths": dict(self.extra_paths),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunArtifacts:
        """Deserialize artifact references from JSON-friendly data."""
        return cls(
            run_json_path=(
                str(data["run_json_path"]) if data.get("run_json_path") is not None else None
            ),
            extra_paths=_copy_str_dict(data.get("extra_paths")),
        )


@dataclass(slots=True)
class RunMetadata:
    """Descriptive metadata captured for one profiling run."""

    run_id: str
    scenario_name: str
    started_at_utc: str
    finished_at_utc: str | None
    hostname: str | None
    python_version: str
    platform: str
    git_commit: str | None
    cwd: str
    command: list[str]
    notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize run metadata to JSON-friendly data."""
        return {
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "hostname": self.hostname,
            "python_version": self.python_version,
            "platform": self.platform,
            "git_commit": self.git_commit,
            "cwd": self.cwd,
            "command": list(self.command),
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunMetadata:
        """Deserialize run metadata from JSON-friendly data."""
        finished_raw = data.get("finished_at_utc")
        hostname_raw = data.get("hostname")
        git_commit_raw = data.get("git_commit")
        return cls(
            run_id=str(data["run_id"]),
            scenario_name=str(data["scenario_name"]),
            started_at_utc=str(data["started_at_utc"]),
            finished_at_utc=str(finished_raw) if finished_raw is not None else None,
            hostname=str(hostname_raw) if hostname_raw is not None else None,
            python_version=str(data["python_version"]),
            platform=str(data["platform"]),
            git_commit=str(git_commit_raw) if git_commit_raw is not None else None,
            cwd=str(data["cwd"]),
            command=_copy_str_list(data.get("command")),
            notes=_copy_str_dict(data.get("notes")),
        )


@dataclass(slots=True)
class RunTimingSummary:
    """Top-level timing summary for one profiling run."""

    wall_time_seconds: float

    def to_dict(self) -> dict[str, float]:
        """Serialize timing data to JSON-friendly data."""
        return {"wall_time_seconds": self.wall_time_seconds}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunTimingSummary:
        """Deserialize timing data from JSON-friendly data."""
        return cls(wall_time_seconds=float(data["wall_time_seconds"]))


@dataclass(slots=True)
class RunResult:
    """Top-level persisted result for one profiling run."""

    status: RunStatus
    metadata: RunMetadata
    timing: RunTimingSummary
    artifacts: RunArtifacts
    error_message: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete run result."""
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "timing": self.timing.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RunResult:
        """Deserialize the complete run result."""
        metadata_raw = data.get("metadata")
        timing_raw = data.get("timing")
        artifacts_raw = data.get("artifacts")
        if not isinstance(metadata_raw, Mapping):
            raise TypeError("RunResult.metadata must be a mapping")
        if not isinstance(timing_raw, Mapping):
            raise TypeError("RunResult.timing must be a mapping")
        if not isinstance(artifacts_raw, Mapping):
            raise TypeError("RunResult.artifacts must be a mapping")

        error_raw = data.get("error_message")
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            status=RunStatus(str(data["status"])),
            metadata=RunMetadata.from_dict(metadata_raw),
            timing=RunTimingSummary.from_dict(timing_raw),
            artifacts=RunArtifacts.from_dict(artifacts_raw),
            error_message=str(error_raw) if error_raw is not None else None,
        )


def run_result_to_dict(run_result: RunResult) -> dict[str, object]:
    """Serialize a run result using the stable PR1 artifact schema."""
    return run_result.to_dict()
