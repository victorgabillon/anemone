"""Lightweight scenario registry for standalone profiling runs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .scenario_runtime import ScenarioRuntime, ScenarioRuntimeOptions


@dataclass(slots=True)
class ProfilingScenario:
    """One runnable profiling scenario."""

    name: str
    description: str
    runner: Callable[[], None]
    runtime_builder: Callable[[ScenarioRuntimeOptions], ScenarioRuntime] | None = None


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
        raise _lazy_builder_type_error(scenario)
    if scenario.name != registration.name:
        raise _lazy_builder_name_error(registration.name, scenario.name)

    register_scenario(scenario)
    return scenario


def get_scenario(name: str) -> ProfilingScenario:
    """Return a registered profiling scenario by name."""
    try:
        registered = _SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_SCENARIOS))
        raise _unknown_scenario_error(name, available) from exc

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
    name="cheap_eval",
    description=(
        "Synthetic deterministic scenario with minimal evaluator cost to expose "
        "framework overhead."
    ),
    loader_module="anemone.profiling.scenario_cheap_eval",
    loader_attribute="build_cheap_eval_scenario",
)

register_lazy_scenario(
    name="deep_tree",
    description="Synthetic deterministic scenario with deeper narrow structure.",
    loader_module="anemone.profiling.scenario_deep_tree",
    loader_attribute="build_deep_tree_scenario",
)

register_lazy_scenario(
    name="expensive_eval",
    description="Synthetic deterministic scenario with higher evaluator CPU cost.",
    loader_module="anemone.profiling.scenario_expensive_eval",
    loader_attribute="build_expensive_eval_scenario",
)

register_lazy_scenario(
    name="reuse_heavy",
    description=(
        "Synthetic deterministic scenario with repeated shared states to stress reuse."
    ),
    loader_module="anemone.profiling.scenario_reuse_heavy",
    loader_attribute="build_reuse_heavy_scenario",
)

register_lazy_scenario(
    name="smoke",
    description="Tiny real end-to-end search using public Anemone APIs.",
    loader_module="anemone.profiling.scenario_smoke",
    loader_attribute="build_smoke_scenario",
)

register_lazy_scenario(
    name="wide_tree",
    description="Synthetic deterministic scenario with high branching factor.",
    loader_module="anemone.profiling.scenario_wide_tree",
    loader_attribute="build_wide_tree_scenario",
)


def _lazy_builder_type_error(scenario: object) -> TypeError:
    return TypeError(
        "Lazy profiling scenario builder must return ProfilingScenario, "
        f"got {type(scenario)}"
    )


def _lazy_builder_name_error(expected_name: str, actual_name: str) -> ValueError:
    return ValueError(
        f"Lazy profiling scenario name mismatch: expected {expected_name!r}, "
        f"got {actual_name!r}"
    )


def _unknown_scenario_error(name: str, available: str) -> ValueError:
    return ValueError(
        f"Unknown profiling scenario {name!r}. Available scenarios: {available}"
    )
