"""Run launcher page for the profiling dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

from anemone.profiling.external_profilers import SUPPORTED_EXTERNAL_PROFILERS
from anemone.profiling.gui import get_streamlit
from anemone.profiling.runner import run_scenario
from anemone.profiling.scenarios import list_scenarios
from anemone.profiling.suite_artifacts import SUITE_JSON_FILENAME
from anemone.profiling.suite_runner import run_suite
from anemone.profiling.suites import list_suites


def render_run_launcher(base_dir: Path) -> None:
    """Render scenario and suite launch controls."""
    st = get_streamlit()
    st.title("Run launcher")
    st.caption(f"New artifacts will be written under `{base_dir}`.")

    st.subheader("Run scenario")
    with st.form("profiling_run_scenario_form"):
        scenario_names = [scenario.name for scenario in list_scenarios()]
        scenario_name = st.selectbox("Scenario", options=scenario_names)
        profiler = st.selectbox("Profiler", options=SUPPORTED_EXTERNAL_PROFILERS)
        component_summary = st.checkbox("Component summary", value=False)
        run_submitted = st.form_submit_button("Run scenario")

    if run_submitted:
        with st.spinner("Running scenario..."):
            run_result = run_scenario(
                scenario_name,
                base_dir,
                profiler=profiler,
                component_summary=component_summary,
                command=[sys.executable, "-m", "anemone.profiling.gui"],
            )
        if run_result.artifacts.run_json_path is not None:
            st.success(f"Run artifact written to {run_result.artifacts.run_json_path}")
        if run_result.error_message is not None:
            st.error(run_result.error_message)

    st.subheader("Run suite")
    with st.form("profiling_run_suite_form"):
        suite_names = [suite.name for suite in list_suites()]
        suite_name = st.selectbox("Suite", options=suite_names)
        repetitions = st.number_input(
            "Repetitions",
            min_value=1,
            value=1,
            step=1,
        )
        suite_profiler = st.selectbox(
            "Suite profiler",
            options=SUPPORTED_EXTERNAL_PROFILERS,
            key="profiling_suite_profiler",
        )
        suite_component_summary = st.checkbox(
            "Suite component summary",
            value=False,
        )
        suite_submitted = st.form_submit_button("Run suite")

    if suite_submitted:
        with st.spinner("Running suite..."):
            suite_result = run_suite(
                suite_name,
                base_dir,
                repetitions=int(repetitions),
                profiler=suite_profiler,
                component_summary=suite_component_summary,
                command=[sys.executable, "-m", "anemone.profiling.gui"],
            )
        suite_json_path = Path(base_dir) / suite_result.run_id / SUITE_JSON_FILENAME
        st.success(f"Suite artifact written to {suite_json_path}")
        if suite_result.error_message is not None:
            st.error(suite_result.error_message)
