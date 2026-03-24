"""Standalone profiling foundation for Anemone search runs."""
# pylint: disable=import-outside-toplevel

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
    from .suite_runner import run_suite
    from .suites import ProfilingSuite, get_suite, list_suites

__all__ = [
    "SCHEMA_VERSION",
    "ProfilingScenario",
    "ProfilingSuite",
    "RunArtifacts",
    "RunMetadata",
    "RunResult",
    "RunStatus",
    "RunTimingSummary",
    "get_scenario",
    "get_suite",
    "list_scenarios",
    "list_suites",
    "run_result_to_dict",
    "run_scenario",
    "run_suite",
]


def __getattr__(name: str) -> Any:
    """Lazy-load heavier runtime helpers to keep module entrypoints clean."""
    if name == "run_scenario":
        from .runner import run_scenario as loaded_run_scenario

        return loaded_run_scenario
    if name == "run_suite":
        from .suite_runner import run_suite as loaded_run_suite

        return loaded_run_suite
    if name in {"ProfilingScenario", "get_scenario", "list_scenarios"}:
        from . import scenarios as scenarios_module

        loaded_objects = {
            "ProfilingScenario": scenarios_module.ProfilingScenario,
            "get_scenario": scenarios_module.get_scenario,
            "list_scenarios": scenarios_module.list_scenarios,
        }
        return loaded_objects[name]
    if name in {"ProfilingSuite", "get_suite", "list_suites"}:
        from . import suites as suites_module

        loaded_objects = {
            "ProfilingSuite": suites_module.ProfilingSuite,
            "get_suite": suites_module.get_suite,
            "list_suites": suites_module.list_suites,
        }
        return loaded_objects[name]
    raise _missing_attribute_error(name)


def _missing_attribute_error(name: str) -> AttributeError:
    return AttributeError(f"module {__name__!r} has no attribute {name!r}")
