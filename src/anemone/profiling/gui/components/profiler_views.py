"""Profiler-tool integrations for run detail pages."""

from __future__ import annotations

import shlex
from pathlib import Path

from anemone.profiling.gui import (
    get_streamlit,
    get_streamlit_components_v1,
)
from anemone.profiling.gui.profiler_tools import (
    callgraph_artifact_path,
    detect_tool_availability,
    generate_gprof2dot_dot,
    generate_gprof2dot_svg,
    render_dot_graph,
    snakeviz_command,
)


def render_profiler_tooling_section(
    *,
    run_dir: Path,
    pstats_path: Path | None,
    key_prefix: str,
) -> None:
    """Render the profiler visualization tooling section for one run."""
    st = get_streamlit()
    st.markdown("### Call graph & interactive profile tools")
    if pstats_path is None or not pstats_path.exists():
        st.info("No `cprofile.pstats` artifact is available for this run.")
        return

    availability = detect_tool_availability()
    st.caption(f"`cprofile.pstats`: `{pstats_path}`")
    columns = st.columns(3)
    columns[0].metric("SnakeViz", _availability_label(availability.snakeviz))
    columns[1].metric("gprof2dot", _availability_label(availability.gprof2dot))
    columns[2].metric("Graphviz dot", _availability_label(availability.graphviz_dot))

    render_snakeviz_section(run_dir, pstats_path)
    render_gprof2dot_section(run_dir, pstats_path, key_prefix=key_prefix)


def render_snakeviz_section(run_dir: Path, pstats_path: Path) -> None:
    """Render the SnakeViz helper panel for one ``cProfile`` artifact."""
    del run_dir
    st = get_streamlit()
    availability = detect_tool_availability()
    with st.expander("SnakeViz", expanded=False):
        st.write(
            "SnakeViz is an external browser-based viewer for `cProfile` output. "
            "It is useful when you want interactive call-stack exploration rather "
            "than a flat top-functions table."
        )
        command = snakeviz_command(pstats_path)
        st.code(shlex.join(command), language="bash")
        if availability.snakeviz:
            st.success("`snakeviz` is installed in this environment.")
        else:
            st.warning("`snakeviz` is not installed in this environment.")
        st.caption("Run the command above in your shell to open the profile locally.")


def render_gprof2dot_section(
    run_dir: Path,
    pstats_path: Path,
    *,
    key_prefix: str,
) -> None:
    """Render gprof2dot controls and generated call-graph output."""
    st = get_streamlit()
    availability = detect_tool_availability()
    with st.expander("gprof2dot call graph", expanded=True):
        st.write(
            "gprof2dot converts `cProfile` data into a caller/callee graph. "
            "Use tighter thresholds to simplify the graph when a profile is dense."
        )
        node_threshold = st.slider(
            "Node threshold (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            key=f"{key_prefix}_gprof2dot_node_threshold",
        )
        edge_threshold = st.slider(
            "Edge threshold (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.1,
            step=0.1,
            key=f"{key_prefix}_gprof2dot_edge_threshold",
        )
        output_format = st.selectbox(
            "Output format",
            options=["svg", "png", "dot"],
            key=f"{key_prefix}_gprof2dot_output_format",
        )
        output_path = callgraph_artifact_path(
            run_dir,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            output_format=output_format,
        )
        dot_path = callgraph_artifact_path(
            run_dir,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            output_format="dot",
        )

        _render_gprof2dot_availability(
            gprof2dot_available=availability.gprof2dot,
            graphviz_available=availability.graphviz_dot,
            output_format=output_format,
        )

        action_label = (
            "Regenerate call graph" if output_path.exists() else "Generate call graph"
        )
        generation_disabled = (
            not availability.gprof2dot
            or (output_format in {"svg", "png"} and not availability.graphviz_dot)
        )
        if st.button(
            action_label,
            key=f"{key_prefix}_gprof2dot_generate",
            disabled=generation_disabled,
        ):
            _generate_call_graph(
                pstats_path=pstats_path,
                dot_path=dot_path,
                output_path=output_path,
                output_format=output_format,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )

        if output_path.exists():
            st.success(f"Generated artifact: {output_path}")
            render_callgraph_artifact(output_path)
        else:
            st.caption(
                f"No generated `{output_format}` call graph exists yet for these thresholds."
            )

        if dot_path.exists() and output_format != "dot":
            st.caption(f"Intermediate DOT artifact: `{dot_path}`")


def render_callgraph_artifact(output_path: Path) -> None:
    """Render a generated call graph artifact inline when possible."""
    st = get_streamlit()
    output_format = output_path.suffix.lstrip(".").lower()
    st.caption(f"Artifact path: `{output_path}`")
    if output_format == "svg":
        render_callgraph_svg(output_path)
        return
    if output_format == "png":
        st.image(str(output_path), use_container_width=True)
        return
    if output_format == "dot":
        try:
            st.code(output_path.read_text(encoding="utf-8"), language="dot")
        except OSError:
            st.info("The DOT artifact was generated, but it could not be read back.")
        return
    st.info(f"Generated artifact is available at `{output_path}`.")


def render_callgraph_svg(svg_path: Path) -> None:
    """Render one SVG call graph inline when possible."""
    st = get_streamlit()
    components = get_streamlit_components_v1()
    try:
        svg_text = svg_path.read_text(encoding="utf-8")
    except OSError:
        st.info("The SVG artifact was generated, but it could not be read back.")
        return
    if components is None:
        st.code(svg_text, language="xml")
        return
    components.html(svg_text, height=720, scrolling=True)


def _generate_call_graph(
    *,
    pstats_path: Path,
    dot_path: Path,
    output_path: Path,
    output_format: str,
    node_threshold: float,
    edge_threshold: float,
) -> None:
    st = get_streamlit()
    try:
        if output_format == "svg":
            generate_gprof2dot_svg(
                pstats_path,
                output_path,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
        elif output_format == "png":
            generate_gprof2dot_dot(
                pstats_path,
                dot_path,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
            render_dot_graph(dot_path, output_path, output_format="png")
        elif output_format == "dot":
            generate_gprof2dot_dot(
                pstats_path,
                output_path,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
        else:
            st.error(f"Unsupported output format `{output_format}`.")
            return
    except RuntimeError as exc:
        st.error(str(exc))
        return
    st.success(f"Call graph generated at `{output_path}`.")


def _render_gprof2dot_availability(
    *,
    gprof2dot_available: bool,
    graphviz_available: bool,
    output_format: str,
) -> None:
    st = get_streamlit()
    if not gprof2dot_available:
        st.warning("`gprof2dot` is not installed in this environment.")
        return
    st.success("`gprof2dot` is installed in this environment.")
    if output_format in {"svg", "png"}:
        if graphviz_available:
            st.success("Graphviz `dot` is available for rendered call-graph images.")
        else:
            st.warning(
                "Graphviz `dot` executable was not found; DOT generation is still "
                "possible but SVG/PNG rendering is unavailable."
            )


def _availability_label(is_available: bool) -> str:
    return "Available" if is_available else "Missing"
