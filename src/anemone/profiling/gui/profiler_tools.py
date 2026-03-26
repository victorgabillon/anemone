"""Helpers for optional external profiler-visualization tools."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GraphOutputFormat = Literal["svg", "png", "dot"]
_VISUALIZATION_DIRNAME = "profiler_visualizations"


@dataclass(slots=True)
class ToolAvailability:
    """Availability flags for optional local profiler visualization tools."""

    snakeviz: bool
    gprof2dot: bool
    graphviz_dot: bool


def detect_tool_availability() -> ToolAvailability:
    """Return availability flags for supported optional tools."""
    return ToolAvailability(
        snakeviz=shutil.which("snakeviz") is not None,
        gprof2dot=shutil.which("gprof2dot") is not None,
        graphviz_dot=shutil.which("dot") is not None,
    )


def snakeviz_command(pstats_path: Path) -> list[str]:
    """Return the local SnakeViz command for one ``.pstats`` artifact."""
    snakeviz_binary = shutil.which("snakeviz") or "snakeviz"
    return [snakeviz_binary, str(Path(pstats_path).resolve())]


def profiler_visualizations_dir(run_dir: Path) -> Path:
    """Return the stable directory used for generated GUI visualization artifacts."""
    return Path(run_dir) / _VISUALIZATION_DIRNAME


def callgraph_artifact_path(
    run_dir: Path,
    *,
    node_threshold: float,
    edge_threshold: float,
    output_format: GraphOutputFormat,
) -> Path:
    """Return the deterministic output path for one generated call graph artifact."""
    stem = (
        "callgraph"
        f"_n{_threshold_token(node_threshold)}"
        f"_e{_threshold_token(edge_threshold)}"
    )
    return profiler_visualizations_dir(run_dir) / f"{stem}.{output_format}"


def generate_gprof2dot_dot(
    pstats_path: Path,
    output_dot_path: Path,
    *,
    node_threshold: float,
    edge_threshold: float,
) -> None:
    """Generate a DOT call graph from one ``cProfile`` stats file."""
    gprof2dot_binary = _require_binary("gprof2dot")
    output_path = Path(output_dot_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        gprof2dot_binary,
        "-f",
        "pstats",
        "-n",
        _threshold_argument(node_threshold),
        "-e",
        _threshold_argument(edge_threshold),
        str(Path(pstats_path).resolve()),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime fallback
        raise RuntimeError(
            exc.stderr.strip() or "gprof2dot failed to generate a DOT graph."
        ) from exc
    output_path.write_text(result.stdout, encoding="utf-8")


def generate_gprof2dot_svg(
    pstats_path: Path,
    output_svg_path: Path,
    *,
    node_threshold: float,
    edge_threshold: float,
) -> None:
    """Generate an SVG call graph from one ``cProfile`` stats file."""
    dot_path = Path(output_svg_path).with_suffix(".dot")
    build_gprof2dot_graph(
        pstats_path=pstats_path,
        dot_path=dot_path,
        output_path=output_svg_path,
        output_format="svg",
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )


def build_gprof2dot_graph(
    *,
    pstats_path: Path,
    dot_path: Path,
    output_path: Path,
    output_format: GraphOutputFormat,
    node_threshold: float,
    edge_threshold: float,
) -> None:
    """Generate DOT and optional rendered graph artifacts from cProfile stats."""
    generate_gprof2dot_dot(
        pstats_path,
        dot_path,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    if output_format != "dot":
        render_dot_graph(
            dot_path,
            output_path,
            output_format=output_format,
        )


def render_dot_graph(
    dot_path: Path,
    output_path: Path,
    *,
    output_format: GraphOutputFormat,
) -> None:
    """Render one DOT graph to an image format using Graphviz ``dot``."""
    if output_format == "dot":
        raise _dot_render_not_needed_error()

    dot_binary = _require_binary("dot")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        dot_binary,
        f"-T{output_format}",
        str(Path(dot_path).resolve()),
        "-o",
        str(destination.resolve()),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime fallback
        raise RuntimeError(
            exc.stderr.strip()
            or f"Graphviz dot failed to render a {output_format.upper()} call graph."
        ) from exc


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise _missing_binary_error(name)
    return binary


def _dot_render_not_needed_error() -> ValueError:
    return ValueError("DOT output does not require Graphviz rendering.")


def _missing_binary_error(name: str) -> RuntimeError:
    return RuntimeError(f"`{name}` is not installed in this environment.")


def _threshold_argument(value: float) -> str:
    return f"{value:.2f}"


def _threshold_token(value: float) -> str:
    return _threshold_argument(value).replace(".", "p")
