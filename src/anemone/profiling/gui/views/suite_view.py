"""Suite browser and detail view for the profiling dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.artifacts import RunStatus
from anemone.profiling.gui import get_streamlit
from anemone.profiling.gui.components.charts import (
    plot_repetition_series,
    plot_suite_aggregates,
)
from anemone.profiling.gui.components.selectors import select_suite
from anemone.profiling.gui.components.tables import (
    repetition_rows,
    suite_aggregate_rows,
    suite_rows,
)
from anemone.profiling.gui.data_loading import discover_suites

if TYPE_CHECKING:
    from pathlib import Path


def render_suite_view(base_dir: Path) -> None:
    """Render the suite browser and detail view."""
    st = get_streamlit()
    suites = discover_suites(base_dir)

    st.title("Suites")
    st.caption(f"Discovered suite artifacts under `{base_dir}`.")
    st.dataframe(suite_rows(suites), use_container_width=True, hide_index=True)

    selected_suite = select_suite(
        suites,
        label="Inspect suite",
        key="profiling_selected_suite_run_id",
    )
    if selected_suite is None:
        return

    st.subheader("Suite overview")
    metrics = st.columns(4)
    metrics[0].metric("Suite", selected_suite.suite_name)
    metrics[1].metric("Repetitions", selected_suite.requested_repetitions)
    metrics[2].metric("Profiler", selected_suite.profiler)
    metrics[3].metric("Component summary", str(selected_suite.component_summary))
    st.write(selected_suite.description)

    st.subheader("Scenario aggregates")
    st.dataframe(
        suite_aggregate_rows(selected_suite),
        use_container_width=True,
        hide_index=True,
    )
    plot_suite_aggregates(selected_suite)

    scenario_names = [
        aggregate.scenario_name for aggregate in selected_suite.scenario_aggregates
    ]
    if not scenario_names:
        return

    selected_scenario_name = st.selectbox(
        "Repetition distribution",
        options=scenario_names,
        key="profiling_selected_suite_scenario_name",
    )
    wall_times = [
        repetition.wall_time_seconds
        for repetition in selected_suite.scenario_runs
        if repetition.scenario_name == selected_scenario_name
        and repetition.status is RunStatus.SUCCESS
        and repetition.wall_time_seconds is not None
    ]
    if wall_times:
        plot_repetition_series(wall_times)

    st.dataframe(
        repetition_rows(selected_suite, scenario_name=selected_scenario_name),
        use_container_width=True,
        hide_index=True,
    )
