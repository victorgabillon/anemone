"""Shared helpers for the toy-domain / real-engine example scripts."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

from anemone.debug.toy_scenarios import (
    ToyScenarioSpec,
    all_toy_scenario_specs,
    build_live_toy_debug_environment,
)

if TYPE_CHECKING:
    from anemone.debug import LiveDebugEnvironment

_DEFAULT_OUTPUT_ROOT = Path("runs") / "debug_scenarios"


def run_live_scenario(
    scenario_spec: ToyScenarioSpec,
    *,
    session_directory: str | Path | None = None,
    snapshot_format: str = "svg",
) -> tuple[object, LiveDebugEnvironment]:
    """Run one tiny toy domain through the real Anemone engine."""
    resolved_directory = Path(session_directory or (_DEFAULT_OUTPUT_ROOT / scenario_spec.name))
    debug_environment = build_live_toy_debug_environment(
        scenario_spec,
        resolved_directory,
        snapshot_format=snapshot_format,
    )
    try:
        result = debug_environment.controlled_exploration.explore(
            random_generator=Random(0)
        )
    finally:
        debug_environment.finalize()
    return result, debug_environment


def run_and_report_scenario(
    scenario_spec: ToyScenarioSpec,
    *,
    session_directory: str | Path | None = None,
    snapshot_format: str = "svg",
) -> tuple[object, LiveDebugEnvironment]:
    """Run one toy domain, finalize it, and print the usage summary."""
    result, debug_environment = run_live_scenario(
        scenario_spec,
        session_directory=session_directory,
        snapshot_format=snapshot_format,
    )
    print_scenario_summary(
        scenario_spec,
        session_directory=debug_environment.session_directory,
    )
    print_serving_instructions(debug_environment.session_directory)
    return result, debug_environment


def print_scenario_summary(
    scenario_spec: ToyScenarioSpec,
    *,
    session_directory: str | Path,
) -> None:
    """Print the scenario name, expectations, and session output directory."""
    print(f"Scenario: {scenario_spec.name}")
    print("Execution path: toy domain, real Anemone exploration stack")
    print(scenario_spec.description)
    if scenario_spec.expected_root_value is not None:
        print(f"Expected root value: {scenario_spec.expected_root_value}")
    if scenario_spec.expected_pv:
        print(f"Expected PV: {' -> '.join(scenario_spec.expected_pv)}")
    print(f"Session directory: {Path(session_directory)}")


def print_serving_instructions(session_directory: str | Path, *, port: int = 8000) -> None:
    """Print the recommended command for serving the resulting live session."""
    session_directory_str = str(Path(session_directory))
    python_literal = json.dumps(session_directory_str)
    print("Serve from another terminal with:")
    print(
        'python -c '
        f'"from anemone.debug import serve_live_debug_session; '
        f'serve_live_debug_session({python_literal}, port={port})"'
    )


def scenario_specs_by_name() -> dict[str, ToyScenarioSpec]:
    """Return all built-in toy scenarios keyed by scenario name."""
    return {scenario_spec.name: scenario_spec for scenario_spec in all_toy_scenario_specs()}


__all__ = [
    "print_scenario_summary",
    "print_serving_instructions",
    "run_and_report_scenario",
    "run_live_scenario",
    "scenario_specs_by_name",
]
