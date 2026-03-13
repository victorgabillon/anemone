"""Run a tiny minimax toy domain through the real Anemone engine."""

from __future__ import annotations

from anemone.debug.toy_scenarios import build_minimax_micro_scenario_spec
from examples.debug_scenarios.common import run_and_report_scenario

if __name__ == "__main__":
    run_and_report_scenario(build_minimax_micro_scenario_spec())
