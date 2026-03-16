"""Run the minimax semantic stress toy scenario through the live debug GUI."""

from __future__ import annotations

from anemone.debug.toy_scenarios import build_minimax_semantic_stress_scenario_spec
from examples.debug_scenarios.common import run_and_report_scenario

if __name__ == "__main__":
    run_and_report_scenario(build_minimax_semantic_stress_scenario_spec())
