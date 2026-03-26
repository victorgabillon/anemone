"""Reusable run detail panels for run and suite drill-down views."""

from __future__ import annotations

from pathlib import Path

from anemone.profiling.gui import get_altair, get_pandas, get_streamlit
from anemone.profiling.gui.components.charts import render_component_breakdown
from anemone.profiling.gui.components.profiler_views import (
    render_profiler_tooling_section,
)
from anemone.profiling.gui.data_loading import (
    extract_component_summary,
    read_text_artifact,
    resolve_cprofile_pstats_path,
    run_dir_from_result,
)
from anemone.profiling.gui.profilers.cprofile_parser import parse_cprofile_stats


def render_run_details(run: object, *, key_prefix: str, heading: str) -> None:
    """Render the full detail panel for one selected run."""
    st = get_streamlit()
    st.subheader(heading)
    _render_run_summary(run)

    run_dir = run_dir_from_result(run)
    if run_dir is None:
        st.info("Run artifact directory could not be resolved for deeper drill-down.")
        return

    summary = extract_component_summary(run_dir)
    if summary is not None:
        render_component_breakdown(summary, key_prefix=key_prefix)
    else:
        st.info("No component summary artifact is available for this run.")

    _render_profiler_details(run, key_prefix=key_prefix)
    render_profiler_tooling_section(
        run_dir=run_dir,
        pstats_path=resolve_cprofile_pstats_path(run),
        key_prefix=key_prefix,
    )


def _render_run_summary(run: object) -> None:
    st = get_streamlit()
    details = st.columns(3)
    details[0].metric("Scenario", str(run.metadata.scenario_name))
    details[1].metric("Status", str(run.status.value))
    details[2].metric("Wall time (s)", f"{float(run.timing.wall_time_seconds):.6f}")

    metadata_rows = [
        {"field": "run_id", "value": run.metadata.run_id},
        {"field": "started_at_utc", "value": run.metadata.started_at_utc},
        {
            "field": "finished_at_utc",
            "value": run.metadata.finished_at_utc or "",
        },
        {
            "field": "profiler",
            "value": run.metadata.notes.get("profiler", "none"),
        },
        {"field": "git_commit", "value": run.metadata.git_commit or ""},
        {"field": "cwd", "value": run.metadata.cwd},
    ]
    st.dataframe(metadata_rows, use_container_width=True, hide_index=True)

    if run.metadata.command:
        st.code(" ".join(run.metadata.command), language="bash")
    if run.error_message is not None:
        st.error(run.error_message)


def _render_profiler_details(
    run: object,
    *,
    key_prefix: str,
) -> None:
    st = get_streamlit()
    cprofile_rows = _load_cprofile_rows(run)
    cprofile_text = read_text_artifact(
        run.artifacts.extra_paths.get("cprofile_top_txt"),
        run=run,
    )
    pyinstrument_text = read_text_artifact(
        run.artifacts.extra_paths.get("pyinstrument_txt"),
        run=run,
    )

    if not cprofile_rows and cprofile_text is None and pyinstrument_text is None:
        st.info("No external profiler artifacts are available for this run.")
        return

    st.markdown("### Function-level profiler insight")
    with st.expander("Profiler details", expanded=bool(cprofile_rows)):
        if cprofile_rows:
            min_top_n = 1 if len(cprofile_rows) < 5 else 5
            max_top_n = min(25, len(cprofile_rows))
            default_top_n = min(10, len(cprofile_rows))
            top_n = st.slider(
                "Top functions by cumulative time",
                min_value=min_top_n,
                max_value=max_top_n,
                value=default_top_n,
                key=f"{key_prefix}_cprofile_top_n",
            )
            _render_cprofile_chart(cprofile_rows[:top_n])
            st.dataframe(
                cprofile_rows,
                use_container_width=True,
                hide_index=True,
            )
        if cprofile_text is not None:
            with st.expander("Raw cProfile text"):
                st.code(cprofile_text, language="text")
        if pyinstrument_text is not None:
            with st.expander("Pyinstrument call tree"):
                st.code(pyinstrument_text, language="text")


def _load_cprofile_rows(run: object) -> list[dict[str, object]]:
    pstats_path = resolve_cprofile_pstats_path(run)
    if pstats_path is None or not pstats_path.exists():
        return []
    try:
        return parse_cprofile_stats(pstats_path)
    except (OSError, TypeError, ValueError):
        return []


def _render_cprofile_chart(rows: list[dict[str, object]]) -> None:
    st = get_streamlit()
    pandas_module = get_pandas()
    altair_module = get_altair()
    if pandas_module is None or altair_module is None:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        return

    dataframe = pandas_module.DataFrame(rows)
    chart = (
        altair_module.Chart(dataframe)
        .transform_fold(
            ["cumulative_time_seconds", "self_time_seconds"],
            as_=["metric_name", "seconds"],
        )
        .mark_bar()
        .encode(
            y=altair_module.Y(
                "qualified_name:N",
                sort="-x",
                title="Function",
            ),
            x=altair_module.X("seconds:Q", title="Seconds"),
            color=altair_module.Color(
                "metric_name:N",
                scale=altair_module.Scale(
                    domain=["cumulative_time_seconds", "self_time_seconds"],
                    range=["#0f766e", "#b45309"],
                ),
                title="Metric",
            ),
            tooltip=[
                altair_module.Tooltip("qualified_name:N", title="Function"),
                altair_module.Tooltip("metric_name:N", title="Metric"),
                altair_module.Tooltip("seconds:Q", title="Seconds", format=".6f"),
                altair_module.Tooltip("call_count:Q", title="Calls"),
            ],
        )
        .properties(height=max(240, len(rows) * 28))
    )
    st.altair_chart(chart, use_container_width=True)
