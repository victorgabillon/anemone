"""Deterministic wide-tree profiling scenario."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .scenario_synthetic_common import (
    SyntheticScenarioConfig,
    build_synthetic_scenario,
)

if TYPE_CHECKING:
    from .scenarios import ProfilingScenario

_WIDE_TREE_CONFIG = SyntheticScenarioConfig(
    name="wide_tree",
    description="Synthetic deterministic scenario with high branching factor.",
    branching_factor=8,
    max_depth=3,
    evaluator_mode="cheap",
    evaluator_work_units=0,
    reuse_pattern="tree",
    stopping_branch_limit=24,
    random_seed=13,
)


def build_wide_tree_scenario() -> ProfilingScenario:
    """Build the wide-tree profiling scenario descriptor."""
    return build_synthetic_scenario(_WIDE_TREE_CONFIG)
