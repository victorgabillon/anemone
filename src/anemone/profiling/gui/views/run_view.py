"""Run browser and detail view for the profiling dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.gui import get_streamlit
from anemone.profiling.gui.components.charts import (
    component_breakdown_rows,
    plot_component_breakdown,
)
from anemone.profiling.gui.components.selectors import select_run
from anemone.profiling.gui.components.tables import run_rows
from anemone.profiling.gui.data_loading import (
    discover_runs,
    extract_component_summary,
    read_text_artifact,
    run_dir_from_result,
)

if TYPE_CHECKING:
    from pathlib import Path


def render_run_view(base_dir: Path) -> None:
    """Render the run browser and detail view."""
    st = get_streamlit()
    runs = discover_runs(base_dir)

    st.title("Runs")
    st.caption(f"Discovered run artifacts under `{base_dir}`.")
    st.dataframe(run_rows(runs), use_container_width=True, hide_index=True)

    selected_run = select_run(
        runs,
        label="Inspect run",
        key="profiling_selected_run_id",
    )
    if selected_run is None:
        return

    st.subheader("Run details")
    details = st.columns(3)
    details[0].metric("Scenario", selected_run.metadata.scenario_name)
    details[1].metric("Status", selected_run.status.value)
    details[2].metric(
        "Wall time (s)",
        f"{selected_run.timing.wall_time_seconds:.6f}",
    )

    metadata_rows = [
        {"field": "run_id", "value": selected_run.metadata.run_id},
        {"field": "started_at_utc", "value": selected_run.metadata.started_at_utc},
        {
            "field": "finished_at_utc",
            "value": selected_run.metadata.finished_at_utc or "",
        },
        {
            "field": "profiler",
            "value": selected_run.metadata.notes.get("profiler", "none"),
        },
        {"field": "git_commit", "value": selected_run.metadata.git_commit or ""},
        {"field": "cwd", "value": selected_run.metadata.cwd},
    ]
    st.dataframe(metadata_rows, use_container_width=True, hide_index=True)

    if selected_run.metadata.command:
        st.code(" ".join(selected_run.metadata.command), language="bash")

    run_dir = run_dir_from_result(selected_run)
    if run_dir is None:
        return

    summary = extract_component_summary(run_dir)
    if summary is not None:
        st.subheader("Component breakdown")
        plot_component_breakdown(summary)
        st.dataframe(
            component_breakdown_rows(summary),
            use_container_width=True,
            hide_index=True,
        )

    _render_text_artifact(
        title="cProfile top output",
        text=read_text_artifact(
            selected_run.artifacts.extra_paths.get("cprofile_top_txt"),
            run=selected_run,
        ),
    )
    _render_text_artifact(
        title="pyinstrument output",
        text=read_text_artifact(
            selected_run.artifacts.extra_paths.get("pyinstrument_txt"),
            run=selected_run,
        ),
    )


def _render_text_artifact(*, title: str, text: str | None) -> None:
    if text is None:
        return
    st = get_streamlit()
    st.subheader(title)
    st.text_area(
        title,
        value=text,
        height=280,
        disabled=True,
    )
