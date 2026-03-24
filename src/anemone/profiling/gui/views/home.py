"""Home page for the profiling dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.gui import get_streamlit
from anemone.profiling.gui.components.tables import run_rows, suite_rows
from anemone.profiling.gui.data_loading import discover_runs, discover_suites
from anemone.profiling.gui.state import set_current_page

if TYPE_CHECKING:
    from pathlib import Path


def render_home(base_dir: Path) -> None:
    """Render the dashboard home view."""
    st = get_streamlit()
    runs = discover_runs(base_dir)
    suites = discover_suites(base_dir)

    st.title("Anemone Profiling Dashboard")
    st.caption(f"Browsing profiling artifacts under `{base_dir}`.")

    metrics = st.columns(3)
    metrics[0].metric("Runs", len(runs))
    metrics[1].metric("Suites", len(suites))
    metrics[2].metric(
        "Latest run",
        runs[0].metadata.started_at_utc if runs else "none",
    )

    actions = st.columns(2)
    if actions[0].button("Run scenario", use_container_width=True):
        set_current_page("Run Launcher")
        st.rerun()
    if actions[1].button("Run suite", use_container_width=True):
        set_current_page("Run Launcher")
        st.rerun()

    if runs:
        st.subheader("Recent runs")
        st.dataframe(
            run_rows(runs[:5]),
            use_container_width=True,
            hide_index=True,
        )

    if suites:
        st.subheader("Recent suites")
        st.dataframe(
            suite_rows(suites[:5]),
            use_container_width=True,
            hide_index=True,
        )
