"""Deterministic reuse-heavy profiling scenario."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .scenario_synthetic_common import (
    SyntheticScenarioConfig,
    build_synthetic_scenario,
)

if TYPE_CHECKING:
    from .scenarios import ProfilingScenario

_REUSE_HEAVY_CONFIG = SyntheticScenarioConfig(
    name="reuse_heavy",
    description=(
        "Synthetic deterministic scenario with repeated shared states to stress reuse."
    ),
    branching_factor=4,
    max_depth=4,
    evaluator_mode="cheap",
    evaluator_work_units=0,
    reuse_pattern="shared_last_layer",
    stopping_branch_limit=20,
    random_seed=19,
)


def build_reuse_heavy_scenario() -> ProfilingScenario:
    """Build the reuse-heavy profiling scenario descriptor."""
    return build_synthetic_scenario(_REUSE_HEAVY_CONFIG)
