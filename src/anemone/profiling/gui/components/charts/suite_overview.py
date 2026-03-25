"""Suite-level overview visualizations."""

from __future__ import annotations

from anemone.profiling.gui import get_altair, get_pandas, get_streamlit
from anemone.profiling.gui.metrics import (
    suite_repetition_metric_rows,
    suite_scenario_metric_rows,
    suite_summary_metrics,
)


def render_suite_overview(suite: object, *, key_prefix: str) -> None:
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
            use_container_width=True,
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


def _render_scenario_bar_chart(rows: list[dict[str, object]]) -> None:
    st = get_streamlit()
    pandas_module = get_pandas()
    altair_module = get_altair()
    if pandas_module is None or altair_module is None:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(rows)
    chart = (
        altair_module.Chart(dataframe)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
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
    repetition_rows: list[dict[str, object]],
    *,
    key_prefix: str,
) -> None:
    st = get_streamlit()
    if not repetition_rows:
        st.info("No successful repetitions available for variability analysis.")
        return

    pandas_module = get_pandas()
    altair_module = get_altair()
    if pandas_module is None or altair_module is None:
        st.dataframe(repetition_rows, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(repetition_rows)
    use_box_plot = st.toggle(
        "Use box plot",
        value=True,
        key=f"{key_prefix}_suite_boxplot_toggle",
    )
    if use_box_plot:
        chart = (
            altair_module.Chart(dataframe)
            .mark_boxplot(size=32, color="#b45309")
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

    st.dataframe(repetition_rows, use_container_width=True, hide_index=True)


def _render_heatmap(rows: list[dict[str, object]]) -> None:
    st = get_streamlit()
    pandas_module = get_pandas()
    altair_module = get_altair()
    metrics = [
        "mean_wall_time_seconds",
        "std_wall_time_seconds",
        "min_wall_time_seconds",
        "max_wall_time_seconds",
    ]

    if pandas_module is None or altair_module is None:
        st.dataframe(rows, use_container_width=True, hide_index=True)
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
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _format_seconds(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}s"
