"""Comparison view for runs and suites in the profiling dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.gui import get_pandas, get_streamlit
from anemone.profiling.gui.components.charts import render_comparison_chart
from anemone.profiling.gui.components.layout import render_breadcrumbs
from anemone.profiling.gui.components.selectors import select_run, select_suite
from anemone.profiling.gui.data_loading import (
    discover_runs,
    discover_suites,
    extract_component_summary,
    run_dir_from_result,
)
from anemone.profiling.gui.metrics import component_breakdown_rows, suite_scenario_metric_rows

if TYPE_CHECKING:
    from pathlib import Path

    from anemone.profiling.artifacts import RunResult
    from anemone.profiling.suite_artifacts import SuiteRunResult


def render_comparison_view(base_dir: Path) -> None:
    """Render run-vs-run and suite-vs-suite comparisons."""
    st = get_streamlit()
    st.title("Compare")
    render_breadcrumbs("Profiling", "Compare")

    runs = discover_runs(base_dir)
    st.subheader("Run vs run")
    if len(runs) < 2:
        st.info("At least two runs are needed for run comparison.")
    else:
        columns = st.columns(2)
        with columns[0]:
            run_a = select_run(
                runs,
                label="Run A",
                key="profiling_compare_run_a",
                default_index=0,
            )
        with columns[1]:
            run_b = select_run(
                runs,
                label="Run B",
                key="profiling_compare_run_b",
                default_index=1,
            )
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
            default_index=0,
        )
    with columns[1]:
        suite_b = select_suite(
            suites,
            label="Suite B",
            key="profiling_compare_suite_b",
            default_index=1,
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
    threshold = st.number_input(
        "Regression threshold (seconds)",
        min_value=0.0,
        value=0.05,
        step=0.01,
        key="profiling_compare_run_threshold",
    )

    rows: list[dict[str, object]] = [_comparison_row("Total wall time", delta, run_a.timing.wall_time_seconds, run_b.timing.wall_time_seconds)]

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
            _component_comparison_rows(
                summary_a_rows=component_breakdown_rows(summary_a),
                summary_b_rows=component_breakdown_rows(summary_b),
            )
        )

    regressions = [
        row
        for row in rows
        if float(row["delta_b_minus_a_seconds"]) > float(threshold)
    ]
    if regressions:
        st.error(
            "Potential regression detected in: "
            + ", ".join(str(row["metric"]) for row in regressions)
        )
    else:
        st.success("No compared metric exceeds the regression threshold.")

    render_comparison_chart(
        rows,
        left_label="Run A",
        right_label="Run B",
    )
    _render_comparison_table(rows, threshold=threshold)


def _render_suite_comparison(
    suite_a: SuiteRunResult,
    suite_b: SuiteRunResult,
) -> None:
    st = get_streamlit()
    rows_a = {row["scenario_name"]: row for row in suite_scenario_metric_rows(suite_a)}
    rows_b = {row["scenario_name"]: row for row in suite_scenario_metric_rows(suite_b)}
    scenario_names = sorted(set(rows_a) | set(rows_b))
    rows = [
        {
            "metric": scenario_name,
            "left_seconds": (
                0.0
                if scenario_name not in rows_a
                or rows_a[scenario_name]["mean_wall_time_seconds"] is None
                else float(rows_a[scenario_name]["mean_wall_time_seconds"])
            ),
            "right_seconds": (
                0.0
                if scenario_name not in rows_b
                or rows_b[scenario_name]["mean_wall_time_seconds"] is None
                else float(rows_b[scenario_name]["mean_wall_time_seconds"])
            ),
        }
        for scenario_name in scenario_names
    ]
    normalized_rows = [
        _comparison_row(
            str(row["metric"]),
            float(row["right_seconds"]) - float(row["left_seconds"]),
            float(row["left_seconds"]),
            float(row["right_seconds"]),
        )
        for row in rows
    ]
    render_comparison_chart(
        normalized_rows,
        left_label="Suite A",
        right_label="Suite B",
    )
    _render_comparison_table(normalized_rows, threshold=0.0)

    pandas_module = get_pandas()
    if pandas_module is None:
        return

    dataframe = pandas_module.DataFrame(
        [
            {
                "scenario_name": row["metric"],
                "suite_a": row["left_seconds"],
                "suite_b": row["right_seconds"],
            }
            for row in normalized_rows
        ]
    ).set_index("scenario_name")
    st.bar_chart(dataframe)


def _component_comparison_rows(
    *,
    summary_a_rows: list[dict[str, object]],
    summary_b_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows_b_by_component = {
        str(row["component"]): row for row in summary_b_rows
    }
    return [
        _comparison_row(
            component_name,
            float(rows_b_by_component[component_name]["wall_time_seconds"])
            - float(row["wall_time_seconds"]),
            float(row["wall_time_seconds"]),
            float(rows_b_by_component[component_name]["wall_time_seconds"]),
        )
        for component_name, row in (
            (str(summary_row["component"]), summary_row) for summary_row in summary_a_rows
        )
        if component_name in rows_b_by_component
    ]


def _comparison_row(
    metric: str,
    delta_seconds: float,
    left_seconds: float,
    right_seconds: float,
) -> dict[str, object]:
    return {
        "metric": metric,
        "left_seconds": left_seconds,
        "right_seconds": right_seconds,
        "delta_b_minus_a_seconds": delta_seconds,
        "percent_change": _percent_change(left_seconds, right_seconds),
    }


def _render_comparison_table(
    rows: list[dict[str, object]],
    *,
    threshold: float,
) -> None:
    st = get_streamlit()
    pandas_module = get_pandas()
    if pandas_module is None:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(rows)
    styled = dataframe.style.apply(
        lambda series: [
            (
                "background-color: rgba(185, 28, 28, 0.18)"
                if series.name == "delta_b_minus_a_seconds"
                and float(value) > threshold
                else ""
            )
            for value in series
        ],
        axis=0,
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _percent_change(left_seconds: float, right_seconds: float) -> float | None:
    if left_seconds == 0.0:
        return None
    return ((right_seconds - left_seconds) / left_seconds) * 100.0
