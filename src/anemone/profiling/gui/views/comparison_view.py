"""Comparison view for runs and suites in the profiling dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.gui import get_pandas, get_streamlit
from anemone.profiling.gui.components.selectors import select_run, select_suite
from anemone.profiling.gui.data_loading import (
    discover_runs,
    discover_suites,
    extract_component_summary,
    run_dir_from_result,
)

if TYPE_CHECKING:
    from pathlib import Path

    from anemone.profiling.artifacts import RunResult
    from anemone.profiling.suite_artifacts import SuiteRunResult


def render_comparison_view(base_dir: Path) -> None:
    """Render run-vs-run and suite-vs-suite comparisons."""
    st = get_streamlit()
    st.title("Compare")

    runs = discover_runs(base_dir)
    st.subheader("Run vs run")
    if len(runs) < 2:
        st.info("At least two runs are needed for run comparison.")
    else:
        columns = st.columns(2)
        with columns[0]:
            run_a = select_run(runs, label="Run A", key="profiling_compare_run_a")
        with columns[1]:
            run_b = select_run(runs, label="Run B", key="profiling_compare_run_b")
        if run_a is not None and run_b is not None:
            _render_run_comparison(run_a, run_b)

    suites = discover_suites(base_dir)
    st.subheader("Suite vs suite")
    if len(suites) < 2:
        st.info("At least two suites are needed for suite comparison.")
        return

    columns = st.columns(2)
    with columns[0]:
        suite_a = select_suite(
            suites,
            label="Suite A",
            key="profiling_compare_suite_a",
        )
    with columns[1]:
        suite_b = select_suite(
            suites,
            label="Suite B",
            key="profiling_compare_suite_b",
        )
    if suite_a is not None and suite_b is not None:
        _render_suite_comparison(suite_a, suite_b)


def _render_run_comparison(run_a: RunResult, run_b: RunResult) -> None:
    st = get_streamlit()
    delta = run_b.timing.wall_time_seconds - run_a.timing.wall_time_seconds
    metrics = st.columns(3)
    metrics[0].metric("Run A wall time (s)", f"{run_a.timing.wall_time_seconds:.6f}")
    metrics[1].metric("Run B wall time (s)", f"{run_b.timing.wall_time_seconds:.6f}")
    metrics[2].metric("Delta (B - A)", f"{delta:.6f}")

    rows: list[dict[str, object]] = [
        {
            "metric": "total_wall_time_seconds",
            "run_a": run_a.timing.wall_time_seconds,
            "run_b": run_b.timing.wall_time_seconds,
            "delta_b_minus_a": delta,
        }
    ]

    summary_a = (
        extract_component_summary(run_dir_a)
        if (run_dir_a := run_dir_from_result(run_a)) is not None
        else None
    )
    summary_b = (
        extract_component_summary(run_dir_b)
        if (run_dir_b := run_dir_from_result(run_b)) is not None
        else None
    )
    if summary_a is not None and summary_b is not None:
        rows.extend(
            [
                _component_delta_row(
                    "evaluator",
                    0.0
                    if summary_a.evaluator is None
                    else summary_a.evaluator.total_wall_time_seconds,
                    0.0
                    if summary_b.evaluator is None
                    else summary_b.evaluator.total_wall_time_seconds,
                ),
                _component_delta_row(
                    "dynamics_step",
                    0.0
                    if summary_a.dynamics_step is None
                    else summary_a.dynamics_step.total_wall_time_seconds,
                    0.0
                    if summary_b.dynamics_step is None
                    else summary_b.dynamics_step.total_wall_time_seconds,
                ),
                _component_delta_row(
                    "dynamics_legal_actions",
                    0.0
                    if summary_a.dynamics_legal_actions is None
                    else summary_a.dynamics_legal_actions.total_wall_time_seconds,
                    0.0
                    if summary_b.dynamics_legal_actions is None
                    else summary_b.dynamics_legal_actions.total_wall_time_seconds,
                ),
                _component_delta_row(
                    "residual_framework",
                    summary_a.residual_framework_wall_time_seconds or 0.0,
                    summary_b.residual_framework_wall_time_seconds or 0.0,
                ),
            ]
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_suite_comparison(
    suite_a: SuiteRunResult,
    suite_b: SuiteRunResult,
) -> None:
    st = get_streamlit()
    aggregates_a = {
        aggregate.scenario_name: aggregate for aggregate in suite_a.scenario_aggregates
    }
    aggregates_b = {
        aggregate.scenario_name: aggregate for aggregate in suite_b.scenario_aggregates
    }
    scenario_names = sorted(set(aggregates_a) | set(aggregates_b))
    rows = [
        {
            "scenario_name": scenario_name,
            "suite_a_mean_seconds": (
                None
                if scenario_name not in aggregates_a
                else aggregates_a[scenario_name].wall_time_mean_seconds
            ),
            "suite_b_mean_seconds": (
                None
                if scenario_name not in aggregates_b
                else aggregates_b[scenario_name].wall_time_mean_seconds
            ),
            "delta_b_minus_a": _delta_or_none(
                None
                if scenario_name not in aggregates_a
                else aggregates_a[scenario_name].wall_time_mean_seconds,
                None
                if scenario_name not in aggregates_b
                else aggregates_b[scenario_name].wall_time_mean_seconds,
            ),
        }
        for scenario_name in scenario_names
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    pandas_module = get_pandas()
    if pandas_module is None:
        return

    dataframe = pandas_module.DataFrame(
        [
            {
                "scenario_name": row["scenario_name"],
                "suite_a": row["suite_a_mean_seconds"] or 0.0,
                "suite_b": row["suite_b_mean_seconds"] or 0.0,
            }
            for row in rows
        ]
    ).set_index("scenario_name")
    st.bar_chart(dataframe)


def _component_delta_row(
    name: str, value_a: float, value_b: float
) -> dict[str, object]:
    return {
        "metric": name,
        "run_a": value_a,
        "run_b": value_b,
        "delta_b_minus_a": value_b - value_a,
    }


def _delta_or_none(value_a: float | None, value_b: float | None) -> float | None:
    if value_a is None or value_b is None:
        return None
    return value_b - value_a
