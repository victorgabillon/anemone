"""Registry and launcher helpers for built-in toy debug scenarios."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random

from anemone.utils.logger import anemone_logger

from ._lazy_builders import build_live_toy_debug_environment
from .model import ToyScenarioSpec
from .scenarios import (
    build_deceptive_trap_scenario_spec,
    build_minimax_micro_scenario_spec,
    build_minimax_semantic_stress_scenario_spec,
    build_single_agent_backup_scenario_spec,
)

type ToyScenarioBuilder = Callable[[], ToyScenarioSpec]

_DEFAULT_BROWSER_SESSION_ROOT = Path("runs") / "debug_browser" / "sessions"


@dataclass(frozen=True, slots=True)
class ScenarioRegistryEntry:
    """Declarative metadata for one built-in launchable toy scenario."""

    name: str
    label: str
    description: str
    builder: ToyScenarioBuilder

    def build_spec(self) -> ToyScenarioSpec:
        """Build and return the concrete toy scenario spec."""
        return self.builder()


@dataclass(frozen=True, slots=True)
class ScenarioRunResult:
    """Structured result returned after one browser-triggered scenario run."""

    ok: bool
    scenario_name: str
    session_directory: str | None
    session_path: str | None
    message: str
    error_code: str | None = None


_REGISTERED_SCENARIOS: tuple[ScenarioRegistryEntry, ...] = (
    ScenarioRegistryEntry(
        name="single_agent_backup",
        label="Single Agent Backup",
        description="Single-agent max backup where the best line should resolve to A -> A2.",
        builder=build_single_agent_backup_scenario_spec,
    ),
    ScenarioRegistryEntry(
        name="minimax_micro",
        label="Minimax Micro",
        description=(
            "Tiny minimax tree where both children are MIN nodes and the root "
            "should prefer A -> A1."
        ),
        builder=build_minimax_micro_scenario_spec,
    ),
    ScenarioRegistryEntry(
        name="deceptive_trap",
        label="Deceptive Trap",
        description=(
            "Deceptive shallow heuristic where branch A starts attractive but "
            "deeper expansion reveals branch B as the final best line."
        ),
        builder=build_deceptive_trap_scenario_spec,
    ),
    ScenarioRegistryEntry(
        name="minimax_semantic_stress",
        label="Minimax Semantic Stress",
        description=(
            "Minimax stress scenario for certainty semantics. Shows TERMINAL "
            "leaves, FORCED solved internal MIN nodes, and ESTIMATE partial "
            "MIN nodes in one GUI tree."
        ),
        builder=build_minimax_semantic_stress_scenario_spec,
    ),
)


def list_registered_scenarios() -> tuple[ScenarioRegistryEntry, ...]:
    """Return the curated set of browser-launchable toy scenarios."""
    return _REGISTERED_SCENARIOS


def get_registered_scenario(name: str) -> ScenarioRegistryEntry | None:
    """Return the named registered scenario, or ``None`` when unknown."""
    for scenario_entry in _REGISTERED_SCENARIOS:
        if scenario_entry.name == name:
            return scenario_entry
    return None


def serialize_registered_scenarios() -> list[dict[str, str]]:
    """Return JSON-friendly metadata for all registered scenarios."""
    return [
        {
            "name": scenario_entry.name,
            "label": scenario_entry.label,
            "description": scenario_entry.description,
        }
        for scenario_entry in _REGISTERED_SCENARIOS
    ]


def run_registered_scenario(
    scenario_name: str,
    *,
    output_root_directory: str | Path = _DEFAULT_BROWSER_SESSION_ROOT,
    snapshot_format: str = "svg",
) -> ScenarioRunResult:
    """Run one registered scenario into a stable browser-visible session directory."""
    scenario_entry = get_registered_scenario(scenario_name)
    if scenario_entry is None:
        return ScenarioRunResult(
            ok=False,
            scenario_name=scenario_name,
            session_directory=None,
            session_path=None,
            message=f"Unknown scenario: {scenario_name}",
            error_code="unknown_scenario",
        )

    session_root_directory = Path(output_root_directory)
    session_directory = session_root_directory / scenario_entry.name
    session_path = f"/sessions/{scenario_entry.name}/"

    try:
        session_root_directory.mkdir(parents=True, exist_ok=True)
        _reset_scenario_session_directory(session_directory)

        scenario_spec = scenario_entry.build_spec()
        anemone_logger.info(
            "Launching toy scenario %s into %s",
            scenario_entry.name,
            session_directory,
        )
        debug_environment = build_live_toy_debug_environment(
            scenario_spec,
            session_directory,
            snapshot_format=snapshot_format,
        )
        try:
            debug_environment.controlled_exploration.explore(random_generator=Random(0))
        finally:
            debug_environment.finalize()
    except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover
        anemone_logger.exception(
            "Failed to launch toy scenario %s into %s",
            scenario_entry.name,
            session_directory,
        )
        return ScenarioRunResult(
            ok=False,
            scenario_name=scenario_entry.name,
            session_directory=str(session_directory),
            session_path=session_path,
            message=f"Failed to run scenario {scenario_entry.name}.",
            error_code="run_failed",
        )

    anemone_logger.info(
        "Launched toy scenario %s successfully into %s",
        scenario_entry.name,
        session_directory,
    )
    return ScenarioRunResult(
        ok=True,
        scenario_name=scenario_entry.name,
        session_directory=str(session_directory),
        session_path=session_path,
        message=f"Loaded {scenario_entry.name}",
    )


def _reset_scenario_session_directory(session_directory: Path) -> None:
    """Remove any prior stable session directory before a rerun."""
    if session_directory.exists():
        shutil.rmtree(session_directory)


__all__ = [
    "ScenarioRegistryEntry",
    "ScenarioRunResult",
    "get_registered_scenario",
    "list_registered_scenarios",
    "run_registered_scenario",
    "serialize_registered_scenarios",
]
