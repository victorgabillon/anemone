"""Suite-level overview visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.gui import get_pandas, get_streamlit
from anemone.profiling.gui.components.chart_utils import (
    build_chart,
    get_chart_modules,
    render_table_fallback,
)
from anemone.profiling.gui.metrics import (
    SuiteRepetitionMetricRow,
    SuiteScenarioMetricRow,
    suite_repetition_metric_rows,
    suite_scenario_metric_rows,
    suite_summary_metrics,
)

if TYPE_CHECKING:
    from anemone.profiling.suite_artifacts import SuiteRunResult


def render_suite_overview(suite: SuiteRunResult, *, key_prefix: str) -> None:
    """Render high-signal suite charts and summary metrics."""
    st = get_streamlit()
    summary = suite_summary_metrics(suite)
    rows = suite_scenario_metric_rows(suite)
    repetition_rows = suite_repetition_metric_rows(suite)

    st.subheader("Suite performance overview")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Scenarios", str(summary["scenario_count"]))
    metric_columns[1].metric(
        "Mean of means",
        _format_seconds(summary["mean_of_means_seconds"]),
    )
    metric_columns[2].metric(
        "Fastest scenario",
        str(summary["fastest_scenario_name"] or "n/a"),
    )
    metric_columns[3].metric(
        "Slowest scenario",
        str(summary["slowest_scenario_name"] or "n/a"),
    )

    if not rows:
        st.info("No suite scenario aggregates are available yet.")
        return

    chart_columns = st.columns(2)
    with chart_columns[0]:
        _render_scenario_bar_chart(rows)
    with chart_columns[1]:
        _render_variability_chart(repetition_rows, key_prefix=key_prefix)

    st.markdown("### Scenario heatmap")
    _render_heatmap(rows)


def plot_repetition_series(values: list[float]) -> None:
    """Render a repetition series chart for one selected scenario."""
    st = get_streamlit()
    pandas_module = get_pandas()
    if pandas_module is None:
        st.dataframe(
            [
                {
                    "repetition_index": index,
                    "wall_time_seconds": value,
                }
                for index, value in enumerate(values, start=1)
            ],
            width="stretch",
            hide_index=True,
        )
        return

    dataframe = pandas_module.DataFrame(
        [
            {
                "repetition_index": index,
                "wall_time_seconds": value,
            }
            for index, value in enumerate(values, start=1)
        ]
    ).set_index("repetition_index")
    st.line_chart(dataframe)


def _render_scenario_bar_chart(rows: list[SuiteScenarioMetricRow]) -> None:
    if (chart_context := build_chart(rows)) is None:
        return
    st, altair_module, chart = chart_context
    chart = (
        chart.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            y=altair_module.Y(
                "scenario_name:N",
                sort="-x",
                title="Scenario",
            ),
            x=altair_module.X(
                "mean_wall_time_seconds:Q",
                title="Mean wall time (s)",
            ),
            color=altair_module.Color(
                "mean_wall_time_seconds:Q",
                scale=altair_module.Scale(scheme="oranges"),
                legend=None,
            ),
            tooltip=[
                altair_module.Tooltip("scenario_name:N", title="Scenario"),
                altair_module.Tooltip(
                    "mean_wall_time_seconds:Q",
                    title="Mean (s)",
                    format=".6f",
                ),
                altair_module.Tooltip(
                    "std_wall_time_seconds:Q",
                    title="Std (s)",
                    format=".6f",
                ),
                altair_module.Tooltip(
                    "min_wall_time_seconds:Q",
                    title="Min (s)",
                    format=".6f",
                ),
                altair_module.Tooltip(
                    "max_wall_time_seconds:Q",
                    title="Max (s)",
                    format=".6f",
                ),
            ],
        )
        .properties(height=max(220, len(rows) * 44), title="Scenario comparison")
    )
    st.altair_chart(chart, use_container_width=True)


def _render_variability_chart(
    repetition_rows: list[SuiteRepetitionMetricRow],
    *,
    key_prefix: str,
) -> None:
    st = get_streamlit()
    if not repetition_rows:
        st.info("No successful repetitions available for variability analysis.")
        return

    use_box_plot = st.toggle(
        "Use box plot",
        value=True,
        key=f"{key_prefix}_suite_boxplot_toggle",
    )
    if (chart_context := build_chart(repetition_rows)) is None:
        return
    _unused_st, altair_module, chart = chart_context
    if use_box_plot:
        chart = (
            chart.mark_boxplot(size=32, color="#b45309")
            .encode(
                x=altair_module.X("scenario_name:N", title="Scenario"),
                y=altair_module.Y(
                    "wall_time_seconds:Q",
                    title="Wall time (s)",
                ),
                tooltip=[
                    altair_module.Tooltip("scenario_name:N", title="Scenario"),
                    altair_module.Tooltip(
                        "wall_time_seconds:Q",
                        title="Wall time (s)",
                        format=".6f",
                    ),
                ],
            )
            .properties(height=280, title="Repetition variability")
        )
        st.altair_chart(chart, use_container_width=True)
        return

    st.dataframe(repetition_rows, width="stretch", hide_index=True)


def _render_heatmap(rows: list[SuiteScenarioMetricRow]) -> None:
    st, pandas_module, altair_module = get_chart_modules()
    metrics = [
        "mean_wall_time_seconds",
        "std_wall_time_seconds",
        "min_wall_time_seconds",
        "max_wall_time_seconds",
    ]

    if pandas_module is None or altair_module is None:
        render_table_fallback(rows)
        return

    dataframe = pandas_module.DataFrame(rows)
    heatmap_source = dataframe[["scenario_name", *metrics]].melt(
        id_vars="scenario_name",
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    chart = (
        altair_module.Chart(heatmap_source)
        .mark_rect(cornerRadius=4)
        .encode(
            x=altair_module.X("metric:N", title=None),
            y=altair_module.Y("scenario_name:N", title=None),
            color=altair_module.Color(
                "value:Q",
                scale=altair_module.Scale(scheme="goldred"),
                title="Seconds",
            ),
            tooltip=[
                altair_module.Tooltip("scenario_name:N", title="Scenario"),
                altair_module.Tooltip("metric:N", title="Metric"),
                altair_module.Tooltip("value:Q", title="Seconds", format=".6f"),
            ],
        )
        .properties(height=max(180, len(rows) * 36))
    )
    st.altair_chart(chart, use_container_width=True)

    styled = dataframe[
        [
            "scenario_name",
            "mean_wall_time_seconds",
            "std_wall_time_seconds",
            "min_wall_time_seconds",
            "max_wall_time_seconds",
            "success_rate",
        ]
    ].style.background_gradient(
        subset=[
            "mean_wall_time_seconds",
            "std_wall_time_seconds",
            "min_wall_time_seconds",
            "max_wall_time_seconds",
        ],
        cmap="OrRd",
    )
    st.dataframe(styled, width="stretch", hide_index=True)


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}s"
