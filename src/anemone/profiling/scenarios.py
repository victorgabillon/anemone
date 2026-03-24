"""Lightweight scenario registry for standalone profiling runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import import_module


@dataclass(slots=True)
class ProfilingScenario:
    """One runnable profiling scenario."""

    name: str
    description: str
    runner: Callable[[], None]


@dataclass(frozen=True, slots=True)
class _LazyScenarioRegistration:
    """Deferred scenario definition loaded only when needed."""

    name: str
    description: str
    loader_module: str
    loader_attribute: str


_SCENARIOS: dict[str, ProfilingScenario | _LazyScenarioRegistration] = {}


def register_scenario(scenario: ProfilingScenario) -> ProfilingScenario:
    """Register an already-materialized profiling scenario."""
    _SCENARIOS[scenario.name] = scenario
    return scenario


def register_lazy_scenario(
    *,
    name: str,
    description: str,
    loader_module: str,
    loader_attribute: str,
) -> None:
    """Register a profiling scenario that will be loaded on first use."""
    _SCENARIOS[name] = _LazyScenarioRegistration(
        name=name,
        description=description,
        loader_module=loader_module,
        loader_attribute=loader_attribute,
    )


def _load_lazy_scenario(
    registration: _LazyScenarioRegistration,
) -> ProfilingScenario:
    module = import_module(registration.loader_module)
    builder = getattr(module, registration.loader_attribute)
    scenario = builder()
    if not isinstance(scenario, ProfilingScenario):
        raise TypeError(
            "Lazy profiling scenario builder must return ProfilingScenario, "
            f"got {type(scenario)}"
        )
    if scenario.name != registration.name:
        raise ValueError(
            "Lazy profiling scenario name mismatch: "
            f"expected {registration.name!r}, got {scenario.name!r}"
        )

    register_scenario(scenario)
    return scenario


def get_scenario(name: str) -> ProfilingScenario:
    """Return a registered profiling scenario by name."""
    try:
        registered = _SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_SCENARIOS))
        raise ValueError(
            f"Unknown profiling scenario {name!r}. Available scenarios: {available}"
        ) from exc

    if isinstance(registered, ProfilingScenario):
        return registered
    return _load_lazy_scenario(registered)


def _run_registered_scenario(name: str) -> None:
    """Load and execute a registered scenario on demand."""
    get_scenario(name).runner()


def list_scenarios() -> list[ProfilingScenario]:
    """Return all registered scenarios sorted by name."""
    scenarios: list[ProfilingScenario] = []
    for name in sorted(_SCENARIOS):
        registered = _SCENARIOS[name]
        if isinstance(registered, ProfilingScenario):
            scenarios.append(registered)
            continue

        scenarios.append(
            ProfilingScenario(
                name=registered.name,
                description=registered.description,
                runner=partial(_run_registered_scenario, registered.name),
            )
        )
    return scenarios


register_lazy_scenario(
    name="smoke",
    description="Tiny real end-to-end search using public Anemone APIs.",
    loader_module="anemone.profiling.scenario_smoke",
    loader_attribute="build_smoke_scenario",
)
