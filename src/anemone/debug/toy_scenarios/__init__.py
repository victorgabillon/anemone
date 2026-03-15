"""Public toy-scenario helpers for the Anemone debug GUI."""

from .builders import build_live_toy_debug_environment, build_toy_tree_exploration
from .model import ToyNodeSpec, ToyPlayerRole, ToyScenarioSpec
from .registry import (
    ScenarioRegistryEntry,
    ScenarioRunResult,
    get_registered_scenario,
    list_registered_scenarios,
    run_registered_scenario,
    serialize_registered_scenarios,
)
from .scenarios import (
    all_toy_scenario_specs,
    build_deceptive_trap_scenario_spec,
    build_minimax_micro_scenario_spec,
    build_minimax_semantic_stress_scenario_spec,
    build_single_agent_backup_scenario_spec,
)

__all__ = [
    "ToyNodeSpec",
    "ToyPlayerRole",
    "ToyScenarioSpec",
    "ScenarioRegistryEntry",
    "ScenarioRunResult",
    "all_toy_scenario_specs",
    "build_deceptive_trap_scenario_spec",
    "build_live_toy_debug_environment",
    "build_minimax_micro_scenario_spec",
    "build_minimax_semantic_stress_scenario_spec",
    "build_single_agent_backup_scenario_spec",
    "build_toy_tree_exploration",
    "get_registered_scenario",
    "list_registered_scenarios",
    "run_registered_scenario",
    "serialize_registered_scenarios",
]
