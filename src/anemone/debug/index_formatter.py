"""Helpers for formatting exploration-index debug lines."""

from __future__ import annotations

from typing import Any

from .formatting import safe_getattr


def format_index_fields(index_data: Any) -> dict[str, str]:
    """Return structured exploration-index fields for ``index_data``."""
    fields: dict[str, str] = {}

    for attribute_name, label in (
        ("index", "index"),
        ("zipf_factored_proba", "zipf_factored_proba"),
        ("min_path_value", "min_path_value"),
        ("max_path_value", "max_path_value"),
        ("max_depth_descendants", "max_depth_descendants"),
    ):
        value = safe_getattr(index_data, attribute_name)
        if value is not None:
            fields[label] = str(value)

    interval = safe_getattr(index_data, "interval")
    if interval is not None:
        min_value = safe_getattr(interval, "min_value")
        max_value = safe_getattr(interval, "max_value")
        if min_value is not None or max_value is not None:
            fields["interval"] = f"[{min_value}, {max_value}]"

    if fields:
        return fields
    return {"index": str(index_data)}


def format_index_lines(index_data: Any) -> list[str]:
    """Return human-readable debug lines for exploration index data."""
    return [f"{key}={value}" for key, value in format_index_fields(index_data).items()]
