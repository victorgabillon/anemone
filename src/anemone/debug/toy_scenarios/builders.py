"""High-level builders for toy domains that run through the real engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anemone.debug.live_setup import LiveDebugEnvironment, build_live_debug_environment

from .adapters import create_real_tree_exploration_for_toy_scenario

if TYPE_CHECKING:
    from pathlib import Path

    from .model import ToyScenarioSpec


def build_toy_tree_exploration(scenario_spec: ToyScenarioSpec) -> Any:
    """Build a real ``TreeExploration`` from one tiny declarative toy domain."""
    return create_real_tree_exploration_for_toy_scenario(scenario_spec)


def build_live_toy_debug_environment(
    scenario_spec: ToyScenarioSpec,
    session_directory: str | Path,
    *,
    snapshot_format: str = "svg",
) -> LiveDebugEnvironment:
    """Build the standard live debug environment around one toy-domain run."""
    toy_tree_exploration = build_toy_tree_exploration(scenario_spec)
    return build_live_debug_environment(
        tree_exploration=toy_tree_exploration,
        session_directory=session_directory,
        snapshot_format=snapshot_format,
    )


__all__ = ["build_live_toy_debug_environment", "build_toy_tree_exploration"]
