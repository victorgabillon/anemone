"""Helpers for extracting readable tables from ``cProfile`` stats files."""

from __future__ import annotations

import pstats
from pathlib import Path
from typing import TypedDict, cast


class CProfileRow(TypedDict):
    """Stable function-level timing row derived from ``pstats`` data."""

    function: str
    location: str
    qualified_name: str
    cumulative_time_seconds: float
    self_time_seconds: float
    call_count: int
    primitive_call_count: int
    filename: str
    line_number: int


StatsKey = tuple[str, int, str]
RawStatsValue = tuple[object, object, object, object, object]


def parse_cprofile_stats(path: Path) -> list[CProfileRow]:
    """Return stable function timing rows sorted by cumulative time."""
    stats = pstats.Stats(str(Path(path)))
    raw_stats = cast("dict[StatsKey, RawStatsValue]", stats.__dict__["stats"])
    rows: list[CProfileRow] = [
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
        for (filename, line_number, function_name), raw_value in raw_stats.items()
        for primitive_calls, total_calls, self_time_seconds, cumulative_time_seconds in [
            (
                _as_int(raw_value[0], field_name="primitive_calls"),
                _as_int(raw_value[1], field_name="total_calls"),
                _as_float(raw_value[2], field_name="self_time_seconds"),
                _as_float(raw_value[3], field_name="cumulative_time_seconds"),
            )
        ]
    ]
    return sorted(
        rows,
        key=lambda row: (
            row["cumulative_time_seconds"],
            row["self_time_seconds"],
        ),
        reverse=True,
    )


def _as_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise _float_field_error(field_name, value)


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise _int_field_error(field_name, value)


def _float_field_error(field_name: str, value: object) -> TypeError:
    return TypeError(f"{field_name} must be float-compatible, got {type(value)}")


def _int_field_error(field_name: str, value: object) -> TypeError:
    return TypeError(f"{field_name} must be int-compatible, got {type(value)}")
