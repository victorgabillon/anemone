"""Deterministic cheap-evaluator profiling scenario."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .scenario_synthetic_common import (
    SyntheticScenarioConfig,
    build_synthetic_scenario,
)

if TYPE_CHECKING:
    from .scenarios import ProfilingScenario

_CHEAP_EVAL_CONFIG = SyntheticScenarioConfig(
    name="cheap_eval",
    description=(
        "Synthetic deterministic scenario with minimal evaluator cost to expose "
        "framework overhead."
    ),
    branching_factor=3,
    max_depth=4,
    evaluator_mode="cheap",
    evaluator_work_units=0,
    reuse_pattern="tree",
    stopping_branch_limit=18,
    random_seed=7,
)


def build_cheap_eval_scenario() -> ProfilingScenario:
    """Build the cheap-evaluator profiling scenario descriptor."""
    return build_synthetic_scenario(_CHEAP_EVAL_CONFIG)
