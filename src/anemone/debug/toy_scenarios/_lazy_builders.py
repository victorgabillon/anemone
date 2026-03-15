"""Lazy accessors for heavy toy-scenario builder functions."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from anemone.debug import LiveDebugEnvironment

    from .model import ToyScenarioSpec


def build_toy_tree_exploration(scenario_spec: ToyScenarioSpec) -> Any:
    """Build a real ``TreeExploration`` from one tiny declarative toy domain."""
    builders_module = import_module("anemone.debug.toy_scenarios.builders")
    build_exploration = builders_module.build_toy_tree_exploration
    return build_exploration(scenario_spec)


def build_live_toy_debug_environment(
    scenario_spec: ToyScenarioSpec,
    session_directory: str | Path,
    *,
    snapshot_format: str = "svg",
) -> LiveDebugEnvironment:
    """Build the standard live debug environment around one toy-domain run."""
    builders_module = import_module("anemone.debug.toy_scenarios.builders")
    build_environment = cast(
        "Callable[..., LiveDebugEnvironment]",
        builders_module.build_live_toy_debug_environment,
    )
    return build_environment(
        scenario_spec,
        session_directory,
        snapshot_format=snapshot_format,
    )


__all__ = [
    "build_live_toy_debug_environment",
    "build_toy_tree_exploration",
]
