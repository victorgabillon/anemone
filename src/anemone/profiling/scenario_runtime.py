"""Helper models for scenarios that can expose wrapped runtime construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .collectors import ComponentCollectors


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeOptions:
    """Options that influence how a profiling scenario builds its runtime."""

    component_summary: bool = False


@dataclass(slots=True)
class ScenarioRuntime:
    """One executable runtime plus any attached collector bundle."""

    runner: Callable[[], None]
    component_collectors: ComponentCollectors | None = None
