"""Streamlit app entrypoint for local profiling exploration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.profiling.gui import get_streamlit
from anemone.profiling.gui.components.selectors import render_base_dir_selector
from anemone.profiling.gui.state import ensure_session_defaults
from anemone.profiling.gui.views.comparison_view import render_comparison_view
from anemone.profiling.gui.views.home import render_home
from anemone.profiling.gui.views.run_launcher import render_run_launcher
from anemone.profiling.gui.views.run_view import render_run_view
from anemone.profiling.gui.views.suite_view import render_suite_view

if TYPE_CHECKING:
    from pathlib import Path

_PAGES = (
    "Home",
    "Runs",
    "Suites",
    "Compare",
    "Run Launcher",
)


def main() -> None:
    """Render the profiling dashboard."""
    st = get_streamlit()
    ensure_session_defaults()
    st.set_page_config(
        page_title="Anemone Profiling",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Anemone Profiling")
    base_dir = render_base_dir_selector()
    current_page = st.sidebar.radio(
        "Page",
        options=_PAGES,
        key="profiling_current_page",
    )
    _render_page(current_page, base_dir)


def _render_page(page_name: str, base_dir: Path) -> None:
    if page_name == "Home":
        render_home(base_dir)
        return
    if page_name == "Runs":
        render_run_view(base_dir)
        return
    if page_name == "Suites":
        render_suite_view(base_dir)
        return
    if page_name == "Compare":
        render_comparison_view(base_dir)
        return
    render_run_launcher(base_dir)


if __name__ == "__main__":
    main()
