"""Comparison view for runs and suites in the profiling dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from anemone.profiling.gui import get_pandas, get_streamlit
from anemone.profiling.gui.components.chart_utils import render_table_fallback
from anemone.profiling.gui.components.charts import render_comparison_chart
from anemone.profiling.gui.components.layout import render_breadcrumbs
from anemone.profiling.gui.components.selectors import select_run, select_suite
from anemone.profiling.gui.data_loading import (
    discover_runs,
    discover_suites,
    extract_component_summary,
    run_dir_from_result,
)
from anemone.profiling.gui.metrics import (
    ComponentBreakdownRow,
    SuiteScenarioMetricRow,
    component_breakdown_rows,
    suite_scenario_metric_rows,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from anemone.profiling.artifacts import RunResult
    from anemone.profiling.gui.components.charts.comparison import ComparisonRow
    from anemone.profiling.suite_artifacts import SuiteRunResult


class _SeriesLike(Protocol):
    """Minimal series protocol needed for DataFrame style callbacks."""

    name: object

    def __iter__(self) -> Iterator[object]: ...


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

    rows: list[ComparisonRow] = [
        _comparison_row(
            "Total wall time",
            delta,
            run_a.timing.wall_time_seconds,
            run_b.timing.wall_time_seconds,
        )
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
            _component_comparison_rows(
                summary_a_rows=component_breakdown_rows(summary_a),
                summary_b_rows=component_breakdown_rows(summary_b),
            )
        )

    regressions = [
        row for row in rows if row["delta_b_minus_a_seconds"] > float(threshold)
    ]
    if regressions:
        st.error(
            "Potential regression detected in: "
            + ", ".join(row["metric"] for row in regressions)
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
    normalized_rows: list[ComparisonRow] = [
        _comparison_row(
            scenario_name,
            _scenario_mean_seconds(rows_b.get(scenario_name))
            - _scenario_mean_seconds(rows_a.get(scenario_name)),
            _scenario_mean_seconds(rows_a.get(scenario_name)),
            _scenario_mean_seconds(rows_b.get(scenario_name)),
        )
        for scenario_name in scenario_names
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
    summary_a_rows: list[ComponentBreakdownRow],
    summary_b_rows: list[ComponentBreakdownRow],
) -> list[ComparisonRow]:
    rows_b_by_component = {row["component"]: row for row in summary_b_rows}
    return [
        _comparison_row(
            component_name,
            rows_b_by_component[component_name]["wall_time_seconds"]
            - row["wall_time_seconds"],
            row["wall_time_seconds"],
            rows_b_by_component[component_name]["wall_time_seconds"],
        )
        for component_name, row in (
            (summary_row["component"], summary_row) for summary_row in summary_a_rows
        )
        if component_name in rows_b_by_component
    ]


def _comparison_row(
    metric: str,
    delta_seconds: float,
    left_seconds: float,
    right_seconds: float,
) -> ComparisonRow:
    return {
        "metric": metric,
        "left_seconds": left_seconds,
        "right_seconds": right_seconds,
        "delta_b_minus_a_seconds": delta_seconds,
        "percent_change": _percent_change(left_seconds, right_seconds),
    }


def _render_comparison_table(
    rows: list[ComparisonRow],
    *,
    threshold: float,
) -> None:
    st = get_streamlit()
    pandas_module = get_pandas()
    if pandas_module is None:
        render_table_fallback(rows)
        return

    dataframe = pandas_module.DataFrame(rows)
    styled = dataframe.style.apply(
        _style_comparison_series_with_threshold(threshold),
        axis=0,
    )
    st.dataframe(styled, width="stretch", hide_index=True)


def _percent_change(left_seconds: float, right_seconds: float) -> float | None:
    if left_seconds == 0.0:
        return None
    return ((right_seconds - left_seconds) / left_seconds) * 100.0


def _scenario_mean_seconds(row: SuiteScenarioMetricRow | None) -> float:
    if row is None or row["mean_wall_time_seconds"] is None:
        return 0.0
    return row["mean_wall_time_seconds"]


def _style_comparison_series_with_threshold(
    threshold: float,
) -> Callable[[_SeriesLike], list[str]]:
    def _inner(series: _SeriesLike) -> list[str]:
        return _style_comparison_series(series, threshold=threshold)

    return _inner


def _style_comparison_series(
    series: _SeriesLike,
    *,
    threshold: float,
) -> list[str]:
    series_name = str(series.name)
    return [
        _comparison_cell_style(
            column_name=series_name,
            value=value,
            threshold=threshold,
        )
        for value in series
    ]


def _comparison_cell_style(
    *,
    column_name: str,
    value: object,
    threshold: float,
) -> str:
    if column_name != "delta_b_minus_a_seconds":
        return ""
    if not isinstance(value, int | float):
        return ""
    return (
        "background-color: rgba(185, 28, 28, 0.18)" if float(value) > threshold else ""
    )
