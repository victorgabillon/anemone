"""Comparison chart helpers."""

from __future__ import annotations

from anemone.profiling.gui import get_altair, get_pandas, get_streamlit


def render_comparison_chart(
    rows: list[dict[str, object]],
    *,
    left_label: str,
    right_label: str,
) -> None:
    """Render a grouped comparison chart for two entities."""
    st = get_streamlit()
    pandas_module = get_pandas()
    altair_module = get_altair()
    if pandas_module is None or altair_module is None:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(
        [
            {
                "metric": str(row["metric"]),
                "run": left_label,
                "seconds": float(row["left_seconds"]),
            }
            for row in rows
        ]
        + [
            {
                "metric": str(row["metric"]),
                "run": right_label,
                "seconds": float(row["right_seconds"]),
            }
            for row in rows
        ]
    )
    chart = (
        altair_module.Chart(dataframe)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=altair_module.X("metric:N", title=None),
            y=altair_module.Y("seconds:Q", title="Seconds"),
            color=altair_module.Color(
                "run:N",
                scale=altair_module.Scale(
                    domain=[left_label, right_label],
                    range=["#0f766e", "#b45309"],
                ),
            ),
            xOffset="run:N",
            tooltip=[
                altair_module.Tooltip("metric:N", title="Metric"),
                altair_module.Tooltip("run:N", title="Series"),
                altair_module.Tooltip("seconds:Q", title="Seconds", format=".6f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)
