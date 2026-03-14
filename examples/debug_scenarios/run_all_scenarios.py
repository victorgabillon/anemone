"""List or run the built-in toy domains through the real Anemone engine."""

from __future__ import annotations

import sys

from examples.debug_scenarios.common import (
    run_and_report_scenario,
    scenario_specs_by_name,
)


def main(argv: list[str]) -> int:
    """Run one named scenario or all scenarios when no name is given."""
    scenarios = scenario_specs_by_name()
    if len(argv) > 2:
        print(
            "Usage: python examples/debug_scenarios/run_all_scenarios.py [scenario-name]"
        )
        print("Available scenarios:", ", ".join(scenarios))
        return 1

    scenario_names = [argv[1]] if len(argv) == 2 else list(scenarios)

    for scenario_name in scenario_names:
        scenario_spec = scenarios.get(scenario_name)
        if scenario_spec is None:
            print(f"Unknown scenario: {scenario_name}")
            print("Available scenarios:", ", ".join(scenarios))
            return 1

        run_and_report_scenario(scenario_spec)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
