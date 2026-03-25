"""Small Streamlit session-state helpers for the profiling dashboard."""

from __future__ import annotations

from pathlib import Path

from anemone.profiling.gui import get_streamlit
from anemone.profiling.storage import default_runs_base_dir

_DEFAULT_STATE: dict[str, object] = {
    "profiling_current_page": "Home",
    "profiling_requested_page": None,
    "profiling_selected_run_id": None,
    "profiling_selected_suite_run_id": None,
    "profiling_compare_run_a": None,
    "profiling_compare_run_b": None,
    "profiling_compare_suite_a": None,
    "profiling_compare_suite_b": None,
}


def ensure_session_defaults() -> None:
    """Populate the dashboard session state with stable default values."""
    session_state = get_streamlit().session_state
    for key, value in _DEFAULT_STATE.items():
        session_state.setdefault(key, value)


def get_base_dir() -> Path:
    """Return the currently selected profiling base directory."""
    ensure_session_defaults()
    session_state = get_streamlit().session_state
    return Path(
        str(
            session_state.get(
                "profiling_base_dir",
                str(default_runs_base_dir()),
            )
        )
    )


def set_base_dir(base_dir: Path) -> None:
    """Update the selected profiling base directory."""
    ensure_session_defaults()
    get_streamlit().session_state["profiling_base_dir"] = str(Path(base_dir))


def set_current_page(page_name: str) -> None:
    """Update the current dashboard page selection."""
    ensure_session_defaults()
    get_streamlit().session_state["profiling_current_page"] = page_name


def request_current_page(page_name: str) -> None:
    """Queue a page change to apply before the page selector widget is rendered."""
    ensure_session_defaults()
    get_streamlit().session_state["profiling_requested_page"] = page_name


def consume_requested_page() -> str | None:
    """Return and clear any queued page change request."""
    ensure_session_defaults()
    session_state = get_streamlit().session_state
    requested_page = session_state.get("profiling_requested_page")
    session_state["profiling_requested_page"] = None
    if requested_page is None:
        return None
    return str(requested_page)
