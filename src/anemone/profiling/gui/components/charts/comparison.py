"""Comparison chart helpers."""

from __future__ import annotations

from typing import TypedDict

from anemone.profiling.gui.components.chart_utils import build_chart


class ComparisonRow(TypedDict):
    """Pairwise comparison row used by grouped charts and tables."""

    metric: str
    left_seconds: float
    right_seconds: float
    delta_b_minus_a_seconds: float
    percent_change: float | None


def render_comparison_chart(
    rows: list[ComparisonRow],
    *,
    left_label: str,
    right_label: str,
) -> None:
    """Render a grouped comparison chart for two entities."""
    chart_rows = [
        {
            "metric": row["metric"],
            "run": left_label,
            "seconds": row["left_seconds"],
        }
        for row in rows
    ] + [
        {
            "metric": row["metric"],
            "run": right_label,
            "seconds": row["right_seconds"],
        }
        for row in rows
    ]
    if (chart_context := build_chart(chart_rows, fallback_rows=rows)) is None:
        return
    st, altair_module, chart = chart_context
    chart = (
        chart.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
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
