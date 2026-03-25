"""Run browser and detail view for the profiling dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.gui import get_streamlit
from anemone.profiling.gui.components.layout import render_breadcrumbs
from anemone.profiling.gui.components.run_details import render_run_details
from anemone.profiling.gui.components.selectors import select_run
from anemone.profiling.gui.components.tables import run_rows
from anemone.profiling.gui.data_loading import discover_runs

if TYPE_CHECKING:
    from pathlib import Path


def render_run_view(base_dir: Path) -> None:
    """Render the run browser and detail view."""
    st = get_streamlit()
    runs = discover_runs(base_dir)

    st.title("Runs")
    render_breadcrumbs("Profiling", "Runs")
    st.caption(f"Discovered run artifacts under `{base_dir}`.")
    st.dataframe(run_rows(runs), use_container_width=True, hide_index=True)

    selected_run = select_run(
        runs,
        label="Inspect run",
        key="profiling_selected_run_id",
    )
    if selected_run is None:
        return

    render_breadcrumbs(
        "Profiling",
        "Runs",
        selected_run.metadata.scenario_name,
        selected_run.metadata.run_id,
    )
    render_run_details(
        selected_run,
        key_prefix=f"run_{selected_run.metadata.run_id}",
        heading="Run details",
    )
