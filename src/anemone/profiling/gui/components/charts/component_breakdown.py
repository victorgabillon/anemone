"""Run-level component breakdown visualizations."""

from __future__ import annotations

from anemone.profiling.gui import get_altair, get_pandas, get_streamlit
from anemone.profiling.gui.metrics import (
    component_breakdown_rows,
    component_detail_rows,
)

_COMPONENT_COLORS = {
    "Evaluator": "#0f766e",
    "Dynamics": "#d97706",
    "Overhead": "#b91c1c",
}


def render_component_breakdown(summary: object, *, key_prefix: str) -> None:
    """Render a richer run-level component breakdown view."""
    st = get_streamlit()
    rows = component_breakdown_rows(summary)
    detail_rows = component_detail_rows(summary)

    st.subheader("Where time goes")
    metric_columns = st.columns(3)
    for column, row in zip(metric_columns, rows, strict=False):
        share_percent = float(row["share_of_total"]) * 100.0
        column.metric(
            str(row["component"]),
            f"{float(row['wall_time_seconds']):.4f}s",
            f"{share_percent:.1f}% of total",
        )

    control_columns = st.columns(2)
    normalize = control_columns[0].toggle(
        "Normalize to % of run",
        value=True,
        key=f"{key_prefix}_component_breakdown_normalize",
    )
    show_pie = control_columns[1].toggle(
        "Show pie view",
        value=False,
        key=f"{key_prefix}_component_breakdown_pie",
    )

    chart_columns = st.columns([2, 1])
    with chart_columns[0]:
        _render_stacked_chart(rows, normalize=normalize)
    with chart_columns[1]:
        if show_pie:
            _render_pie_chart(rows)
        else:
            st.dataframe(detail_rows, use_container_width=True, hide_index=True)

    with st.expander("Detailed component metrics"):
        st.dataframe(detail_rows, use_container_width=True, hide_index=True)


def _render_stacked_chart(
    rows: list[dict[str, object]],
    *,
    normalize: bool,
) -> None:
    st = get_streamlit()
    pandas_module = get_pandas()
    altair_module = get_altair()
    if pandas_module is None or altair_module is None:
        display_rows = [
            {
                "component": row["component"],
                "wall_time_seconds": row["wall_time_seconds"],
                "share_percent": float(row["share_of_total"]) * 100.0,
            }
            for row in rows
        ]
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(
        [
            {
                "view": "Run time",
                "component": row["component"],
                "value": (
                    float(row["share_of_total"]) * 100.0
                    if normalize
                    else float(row["wall_time_seconds"])
                ),
                "wall_time_seconds": float(row["wall_time_seconds"]),
                "share_percent": float(row["share_of_total"]) * 100.0,
            }
            for row in rows
        ]
    )
    title = "Share of total run (%)" if normalize else "Wall time (s)"
    chart = (
        altair_module.Chart(dataframe)
        .mark_bar(size=42)
        .encode(
            y=altair_module.Y("view:N", axis=None),
            x=altair_module.X("value:Q", title=title, stack="zero"),
            color=altair_module.Color(
                "component:N",
                scale=altair_module.Scale(
                    domain=list(_COMPONENT_COLORS),
                    range=list(_COMPONENT_COLORS.values()),
                ),
                legend=altair_module.Legend(title="Component"),
            ),
            tooltip=[
                altair_module.Tooltip("component:N", title="Component"),
                altair_module.Tooltip(
                    "wall_time_seconds:Q",
                    title="Wall time (s)",
                    format=".6f",
                ),
                altair_module.Tooltip(
                    "share_percent:Q",
                    title="Share (%)",
                    format=".2f",
                ),
            ],
        )
        .properties(height=110)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_pie_chart(rows: list[dict[str, object]]) -> None:
    st = get_streamlit()
    pandas_module = get_pandas()
    altair_module = get_altair()
    if pandas_module is None or altair_module is None:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(
        [
            {
                "component": row["component"],
                "wall_time_seconds": float(row["wall_time_seconds"]),
                "share_percent": float(row["share_of_total"]) * 100.0,
            }
            for row in rows
        ]
    )
    chart = (
        altair_module.Chart(dataframe)
        .mark_arc(innerRadius=36)
        .encode(
            theta=altair_module.Theta("wall_time_seconds:Q"),
            color=altair_module.Color(
                "component:N",
                scale=altair_module.Scale(
                    domain=list(_COMPONENT_COLORS),
                    range=list(_COMPONENT_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=[
                altair_module.Tooltip("component:N", title="Component"),
                altair_module.Tooltip(
                    "wall_time_seconds:Q",
                    title="Wall time (s)",
                    format=".6f",
                ),
                altair_module.Tooltip(
                    "share_percent:Q",
                    title="Share (%)",
                    format=".2f",
                ),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)
