"""Standalone profiling foundation for Anemone search runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .artifacts import (
    SCHEMA_VERSION,
    RunArtifacts,
    RunMetadata,
    RunResult,
    RunStatus,
    RunTimingSummary,
    run_result_to_dict,
)

if TYPE_CHECKING:
    from .runner import run_scenario
    from .scenarios import ProfilingScenario, get_scenario, list_scenarios

__all__ = [
    "SCHEMA_VERSION",
    "ProfilingScenario",
    "RunArtifacts",
    "RunMetadata",
    "RunResult",
    "RunStatus",
    "RunTimingSummary",
    "get_scenario",
    "list_scenarios",
    "run_result_to_dict",
    "run_scenario",
]


def __getattr__(name: str) -> Any:
    """Lazy-load heavier runtime helpers to keep module entrypoints clean."""
    if name == "run_scenario":
        from .runner import run_scenario as loaded_run_scenario

        return loaded_run_scenario
    if name in {"ProfilingScenario", "get_scenario", "list_scenarios"}:
        from .scenarios import (
            ProfilingScenario as loaded_profiling_scenario,
            get_scenario as loaded_get_scenario,
            list_scenarios as loaded_list_scenarios,
        )

        loaded_objects = {
            "ProfilingScenario": loaded_profiling_scenario,
            "get_scenario": loaded_get_scenario,
            "list_scenarios": loaded_list_scenarios,
        }
        return loaded_objects[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
