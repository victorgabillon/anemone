"""Helpers for extracting readable tables from ``cProfile`` stats files."""

from __future__ import annotations

import pstats
from pathlib import Path


def parse_cprofile_stats(path: Path) -> list[dict[str, object]]:
    """Return stable function timing rows sorted by cumulative time."""
    stats = pstats.Stats(str(Path(path)))
    rows = [
        {
            "function": function_name,
            "location": f"{Path(filename).name}:{line_number}",
            "qualified_name": (
                f"{function_name} ({Path(filename).name}:{line_number})"
            ),
            "cumulative_time_seconds": cumulative_time_seconds,
            "self_time_seconds": self_time_seconds,
            "call_count": total_calls,
            "primitive_call_count": primitive_calls,
            "filename": filename,
            "line_number": line_number,
        }
        for (filename, line_number, function_name), (
            primitive_calls,
            total_calls,
            self_time_seconds,
            cumulative_time_seconds,
            _callers,
        ) in stats.stats.items()
    ]
    return sorted(
        rows,
        key=lambda row: (
            float(row["cumulative_time_seconds"]),
            float(row["self_time_seconds"]),
        ),
        reverse=True,
    )
