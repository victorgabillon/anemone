"""Small layout helpers for profiling GUI pages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from anemone.profiling.gui import get_streamlit

if TYPE_CHECKING:
    from collections.abc import Sequence


def render_breadcrumbs(*parts: str) -> None:
    """Render a lightweight breadcrumb trail."""
    crumbs = [part for part in parts if part]
    if not crumbs:
        return
    get_streamlit().caption(" / ".join(crumbs))


def render_sidebar_context(*, base_dir: Path, current_page: str) -> None:
    """Render sticky context details in the sidebar."""
    st = get_streamlit()
    session_state = st.session_state
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Context")
    st.sidebar.caption(f"Page: `{current_page}`")
    st.sidebar.caption(f"Base dir: `{Path(base_dir).expanduser()}`")

    _render_sidebar_selection(
        label="Selected run",
        values=_selected_values(
            session_state.get("profiling_selected_run_id"),
            session_state.get("profiling_compare_run_a"),
            session_state.get("profiling_compare_run_b"),
        ),
    )
    _render_sidebar_selection(
        label="Selected suite",
        values=_selected_values(
            session_state.get("profiling_selected_suite_run_id"),
            session_state.get("profiling_compare_suite_a"),
            session_state.get("profiling_compare_suite_b"),
        ),
    )
    selected_scenario = session_state.get("profiling_selected_suite_scenario_name")
    if selected_scenario:
        st.sidebar.caption(f"Scenario focus: `{selected_scenario}`")


def _selected_values(*values: object) -> list[str]:
    unique_values: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in unique_values:
            unique_values.append(text)
    return unique_values


def _render_sidebar_selection(*, label: str, values: Sequence[str]) -> None:
    st = get_streamlit()
    if not values:
        st.sidebar.caption(f"{label}: `none`")
        return
    st.sidebar.caption(f"{label}: `{', '.join(values)}`")
