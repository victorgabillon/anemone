"""Command-line interface for the standalone profiling foundation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .artifacts import RunStatus
from .runner import run_scenario
from .scenarios import list_scenarios
from .storage import default_runs_base_dir


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the top-level profiling CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m anemone.profiling.cli",
        description="Standalone profiling helpers for Anemone runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute one profiling scenario and write a run artifact.",
    )
    run_parser.add_argument(
        "--scenario",
        required=True,
        help="Registered profiling scenario name to execute.",
    )
    run_parser.add_argument(
        "--output-dir",
        default=str(default_runs_base_dir()),
        help="Base directory where the profiling run folder will be created.",
    )

    subparsers.add_parser(
        "list-scenarios",
        help="List available profiling scenarios.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the profiling CLI."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_argument_parser()
    args = parser.parse_args(argv_list)

    if args.command == "list-scenarios":
        for scenario in list_scenarios():
            print(f"{scenario.name}: {scenario.description}")
        return 0

    try:
        result = run_scenario(
            args.scenario,
            Path(args.output_dir),
            command=[sys.executable, "-m", "anemone.profiling.cli", *argv_list],
        )
    except Exception as exc:  # pragma: no cover - simple CLI guard
        print(f"profiling run failed before artifact creation: {exc}", file=sys.stderr)
        return 1

    if result.status is RunStatus.SUCCESS:
        if result.artifacts.run_json_path is not None:
            print(result.artifacts.run_json_path)
        return 0

    if result.artifacts.run_json_path is not None:
        print(result.artifacts.run_json_path, file=sys.stderr)
    if result.error_message is not None:
        print(result.error_message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
