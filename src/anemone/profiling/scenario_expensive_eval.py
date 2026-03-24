"""Deterministic expensive-evaluator profiling scenario."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .scenario_synthetic_common import (
    SyntheticScenarioConfig,
    build_synthetic_scenario,
)

if TYPE_CHECKING:
    from .scenarios import ProfilingScenario

_EXPENSIVE_EVAL_CONFIG = SyntheticScenarioConfig(
    name="expensive_eval",
    description="Synthetic deterministic scenario with higher evaluator CPU cost.",
    branching_factor=3,
    max_depth=4,
    evaluator_mode="expensive",
    evaluator_work_units=3_000,
    reuse_pattern="tree",
    stopping_branch_limit=18,
    random_seed=11,
)


def build_expensive_eval_scenario() -> ProfilingScenario:
    """Build the expensive-evaluator profiling scenario descriptor."""
    return build_synthetic_scenario(_EXPENSIVE_EVAL_CONFIG)
