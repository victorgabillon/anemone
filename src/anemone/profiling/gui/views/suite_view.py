"""Suite browser and detail view for the profiling dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.artifacts import RunStatus
from anemone.profiling.gui import get_streamlit
from anemone.profiling.gui.components.charts import (
    plot_repetition_series,
    render_suite_overview,
)
from anemone.profiling.gui.components.layout import render_breadcrumbs
from anemone.profiling.gui.components.run_details import render_run_details
from anemone.profiling.gui.components.selectors import select_suite
from anemone.profiling.gui.components.tables import (
    repetition_rows,
    suite_aggregate_rows,
    suite_rows,
)
from anemone.profiling.gui.data_loading import (
    discover_suites,
    load_suite_repetition_run,
)

if TYPE_CHECKING:
    from pathlib import Path


def render_suite_view(base_dir: Path) -> None:
    """Render the suite browser and detail view."""
    st = get_streamlit()
    suites = discover_suites(base_dir)

    st.title("Suites")
    render_breadcrumbs("Profiling", "Suites")
    st.caption(f"Discovered suite artifacts under `{base_dir}`.")
    st.dataframe(suite_rows(suites), use_container_width=True, hide_index=True)

    selected_suite = select_suite(
        suites,
        label="Inspect suite",
        key="profiling_selected_suite_run_id",
    )
    if selected_suite is None:
        return

    render_breadcrumbs("Profiling", "Suites", selected_suite.suite_name)
    metrics = st.columns(4)
    metrics[0].metric("Suite", selected_suite.suite_name)
    metrics[1].metric("Repetitions", selected_suite.requested_repetitions)
    metrics[2].metric("Profiler", selected_suite.profiler)
    metrics[3].metric("Component summary", str(selected_suite.component_summary))
    st.write(selected_suite.description)

    render_suite_overview(
        selected_suite,
        key_prefix=f"suite_{selected_suite.run_id}",
    )

    st.subheader("Scenario aggregates")
    st.dataframe(
        suite_aggregate_rows(selected_suite),
        use_container_width=True,
        hide_index=True,
    )

    scenario_names = [
        aggregate.scenario_name for aggregate in selected_suite.scenario_aggregates
    ]
    if not scenario_names:
        return

    if (
        st.session_state.get("profiling_selected_suite_scenario_name")
        not in scenario_names
    ):
        st.session_state["profiling_selected_suite_scenario_name"] = scenario_names[0]
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

    successful_repetitions = [
        repetition
        for repetition in selected_suite.scenario_runs
        if repetition.scenario_name == selected_scenario_name
        and repetition.run_json_path is not None
    ]
    if not successful_repetitions:
        return

    repetition_labels = [
        (
            repetition,
            (
                f"rep {repetition.repetition_index} | "
                f"{repetition.status.value} | "
                f"{'n/a' if repetition.wall_time_seconds is None else f'{repetition.wall_time_seconds:.6f}s'}"
            ),
        )
        for repetition in successful_repetitions
    ]
    repetition_label_options = [label for _, label in repetition_labels]
    if (
        st.session_state.get("profiling_selected_suite_repetition_label")
        not in repetition_label_options
    ):
        st.session_state["profiling_selected_suite_repetition_label"] = (
            repetition_label_options[0]
        )
    selected_repetition_label = st.selectbox(
        "Drill into repetition run",
        options=repetition_label_options,
        key="profiling_selected_suite_repetition_label",
    )
    selected_repetition = next(
        repetition
        for repetition, label in repetition_labels
        if label == selected_repetition_label
    )
    linked_run = load_suite_repetition_run(selected_suite, selected_repetition)
    if linked_run is None:
        st.info("The linked run artifact could not be resolved for this repetition.")
        return

    render_breadcrumbs(
        "Profiling",
        "Suites",
        selected_suite.suite_name,
        selected_scenario_name,
        linked_run.metadata.run_id,
    )
    render_run_details(
        linked_run,
        key_prefix=f"suite_run_{linked_run.metadata.run_id}",
        heading="Linked run details",
    )
