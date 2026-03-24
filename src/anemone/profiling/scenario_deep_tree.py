"""Deterministic deep-tree profiling scenario."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .scenario_synthetic_common import (
    SyntheticScenarioConfig,
    build_synthetic_scenario,
)

if TYPE_CHECKING:
    from .scenarios import ProfilingScenario

_DEEP_TREE_CONFIG = SyntheticScenarioConfig(
    name="deep_tree",
    description="Synthetic deterministic scenario with deeper narrow structure.",
    branching_factor=2,
    max_depth=8,
    evaluator_mode="cheap",
    evaluator_work_units=0,
    reuse_pattern="tree",
    stopping_branch_limit=12,
    random_seed=17,
)


def build_deep_tree_scenario() -> ProfilingScenario:
    """Build the deep-tree profiling scenario descriptor."""
    return build_synthetic_scenario(_DEEP_TREE_CONFIG)
