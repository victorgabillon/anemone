"""Small chart helpers for profiling dashboard views."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.gui import get_pandas, get_streamlit

if TYPE_CHECKING:
    from anemone.profiling.component_summary import ComponentSummary
    from anemone.profiling.suite_artifacts import SuiteRunResult


def plot_component_breakdown(summary: ComponentSummary) -> None:
    """Render a simple bar chart for one component summary."""
    st = get_streamlit()
    records = component_breakdown_rows(summary)
    pandas_module = get_pandas()
    if pandas_module is None:
        st.dataframe(records, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(records).set_index("component")
    st.bar_chart(dataframe)


def component_breakdown_rows(summary: ComponentSummary) -> list[dict[str, object]]:
    """Build stable component rows for tables and charts."""
    return [
        {
            "component": "evaluator",
            "wall_time_seconds": 0.0
            if summary.evaluator is None
            else summary.evaluator.total_wall_time_seconds,
        },
        {
            "component": "dynamics_step",
            "wall_time_seconds": 0.0
            if summary.dynamics_step is None
            else summary.dynamics_step.total_wall_time_seconds,
        },
        {
            "component": "dynamics_legal_actions",
            "wall_time_seconds": 0.0
            if summary.dynamics_legal_actions is None
            else summary.dynamics_legal_actions.total_wall_time_seconds,
        },
        {
            "component": "residual_framework",
            "wall_time_seconds": summary.residual_framework_wall_time_seconds or 0.0,
        },
    ]


def plot_suite_aggregates(suite: SuiteRunResult) -> None:
    """Render mean wall times per scenario for one suite."""
    st = get_streamlit()
    records = [
        {
            "scenario_name": aggregate.scenario_name,
            "wall_time_mean_seconds": aggregate.wall_time_mean_seconds or 0.0,
        }
        for aggregate in suite.scenario_aggregates
    ]
    pandas_module = get_pandas()
    if pandas_module is None:
        st.dataframe(records, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(records).set_index("scenario_name")
    st.bar_chart(dataframe)


def plot_repetition_series(values: list[float]) -> None:
    """Render a simple repetition series chart."""
    st = get_streamlit()
    records = [
        {
            "repetition_index": index,
            "wall_time_seconds": value,
        }
        for index, value in enumerate(values, start=1)
    ]
    pandas_module = get_pandas()
    if pandas_module is None:
        st.dataframe(records, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(records).set_index("repetition_index")
    st.line_chart(dataframe)
