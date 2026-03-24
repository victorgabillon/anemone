"""Command-line interface for the standalone profiling foundation."""
# pylint: disable=broad-exception-caught,duplicate-code

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .artifacts import RunStatus
from .external_profilers import SUPPORTED_EXTERNAL_PROFILERS
from .runner import run_scenario
from .scenarios import list_scenarios
from .storage import default_runs_base_dir
from .suite_artifacts import SUITE_JSON_FILENAME
from .suite_runner import run_suite
from .suites import list_suites

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .external_profilers import ProfilerName


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
    run_parser.add_argument(
        "--profiler",
        choices=SUPPORTED_EXTERNAL_PROFILERS,
        default="none",
        help="Optional external profiler to run around the scenario.",
    )
    run_parser.add_argument(
        "--component-summary",
        action="store_true",
        help="Enable wrapper-based component timing when the scenario supports it.",
    )

    run_suite_parser = subparsers.add_parser(
        "run-suite",
        help="Execute one profiling suite repeatedly and write a suite artifact.",
    )
    run_suite_parser.add_argument(
        "--suite",
        required=True,
        help="Registered profiling suite name to execute.",
    )
    run_suite_parser.add_argument(
        "--output-dir",
        default=str(default_runs_base_dir()),
        help="Base directory where the profiling suite folder will be created.",
    )
    run_suite_parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions to execute per scenario.",
    )
    run_suite_parser.add_argument(
        "--profiler",
        choices=SUPPORTED_EXTERNAL_PROFILERS,
        default="none",
        help="Optional external profiler to run around each scenario repetition.",
    )
    run_suite_parser.add_argument(
        "--component-summary",
        action="store_true",
        help="Enable wrapper-based component timing when the scenarios support it.",
    )

    subparsers.add_parser(
        "list-scenarios",
        help="List available profiling scenarios.",
    )
    subparsers.add_parser(
        "list-suites",
        help="List available profiling suites.",
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

    if args.command == "list-suites":
        for suite in list_suites():
            print(f"{suite.name}: {suite.description}")
        return 0

    if args.command == "run-suite":
        try:
            suite_result = run_suite(
                args.suite,
                Path(args.output_dir),
                repetitions=args.repetitions,
                profiler=cast("ProfilerName", args.profiler),
                component_summary=args.component_summary,
                command=[sys.executable, "-m", "anemone.profiling.cli", *argv_list],
            )
        except Exception as exc:  # pragma: no cover - simple CLI guard
            print(
                f"profiling suite failed before artifact creation: {exc}",
                file=sys.stderr,
            )
            return 1

        suite_json_path = (
            Path(args.output_dir) / suite_result.run_id / SUITE_JSON_FILENAME
        )
        if suite_result.error_message is None:
            print(suite_json_path)
            return 0

        print(suite_json_path, file=sys.stderr)
        print(suite_result.error_message, file=sys.stderr)
        return 1

    try:
        run_result = run_scenario(
            args.scenario,
            Path(args.output_dir),
            command=[sys.executable, "-m", "anemone.profiling.cli", *argv_list],
            profiler=cast("ProfilerName", args.profiler),
            component_summary=args.component_summary,
        )
    except Exception as exc:  # pragma: no cover - simple CLI guard
        print(f"profiling run failed before artifact creation: {exc}", file=sys.stderr)
        return 1

    if run_result.status is RunStatus.SUCCESS:
        if run_result.artifacts.run_json_path is not None:
            print(run_result.artifacts.run_json_path)
        return 0

    if run_result.artifacts.run_json_path is not None:
        print(run_result.artifacts.run_json_path, file=sys.stderr)
    if run_result.error_message is not None:
        print(run_result.error_message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
