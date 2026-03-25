"""Reusable selectors for profiling runs, suites, and base directories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from anemone.profiling.gui import get_streamlit
from anemone.profiling.gui.state import get_base_dir

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anemone.profiling.artifacts import RunResult
    from anemone.profiling.suite_artifacts import SuiteRunResult


def render_base_dir_selector() -> Path:
    """Render the sidebar base-directory selector and return the chosen path."""
    st = get_streamlit()
    current = str(get_base_dir())
    selected_text = st.sidebar.text_input(
        "Profiling directory",
        value=current,
        key="profiling_base_dir",
        help="Base directory used to discover and launch profiling artifacts.",
    )
    selected_path = Path(selected_text).expanduser()
    if not selected_path.exists():
        st.sidebar.warning("The selected directory does not exist yet.")
    return selected_path


def select_run(
    runs: Sequence[RunResult],
    *,
    label: str,
    key: str,
    default_index: int = 0,
) -> RunResult | None:
    """Render a run selector and return the selected run, if any."""
    st = get_streamlit()
    if not runs:
        st.info(
            "No profiling runs yet.\n\n"
            "Run `python -m anemone.profiling.cli run --scenario cheap_eval "
            "--output-dir profiling_runs --component-summary` to create one."
        )
        return None

    options = [run.metadata.run_id for run in runs]
    _initialize_selector_key(st, key=key, options=options, default_index=default_index)

    def _format(run_id: str) -> str:
        return _format_run_option(_run_by_id(runs, run_id))

    selected_id = st.selectbox(
        label,
        options=options,
        key=key,
        format_func=_format,
    )
    return _run_by_id(runs, selected_id)


def select_suite(
    suites: Sequence[SuiteRunResult],
    *,
    label: str,
    key: str,
    default_index: int = 0,
) -> SuiteRunResult | None:
    """Render a suite selector and return the selected suite, if any."""
    st = get_streamlit()
    if not suites:
        st.info(
            "No profiling suites yet.\n\n"
            "Run `python -m anemone.profiling.cli run-suite --suite baseline "
            "--output-dir profiling_runs` to create one."
        )
        return None

    options = [suite.run_id for suite in suites]
    _initialize_selector_key(st, key=key, options=options, default_index=default_index)

    def _format(run_id: str) -> str:
        return _format_suite_option(_suite_by_id(suites, run_id))

    selected_id = st.selectbox(
        label,
        options=options,
        key=key,
        format_func=_format,
    )
    return _suite_by_id(suites, selected_id)


def _run_by_id(runs: Sequence[RunResult], run_id: str) -> RunResult:
    for run in runs:
        if run.metadata.run_id == run_id:
            return run
    raise KeyError(run_id)


def _suite_by_id(suites: Sequence[SuiteRunResult], run_id: str) -> SuiteRunResult:
    for suite in suites:
        if suite.run_id == run_id:
            return suite
    raise KeyError(run_id)


def _format_run_option(run: RunResult) -> str:
    return (
        f"{run.metadata.run_id} | {run.metadata.scenario_name} | "
        f"{run.status.value} | {run.timing.wall_time_seconds:.6f}s"
    )


def _format_suite_option(suite: SuiteRunResult) -> str:
    return (
        f"{suite.run_id} | {suite.suite_name} | "
        f"{suite.requested_repetitions} reps | profiler={suite.profiler}"
    )


def _initialize_selector_key(
    st: object,
    *,
    key: str,
    options: Sequence[str],
    default_index: int,
) -> None:
    session_state = st.session_state
    current_value = session_state.get(key)
    if current_value in options:
        return
    safe_index = min(max(default_index, 0), len(options) - 1)
    session_state[key] = options[safe_index]
