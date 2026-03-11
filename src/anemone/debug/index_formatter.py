"""Helpers for formatting exploration-index debug lines."""

from __future__ import annotations

from typing import Any

from .formatting import safe_getattr


def format_index_lines(index_data: Any) -> list[str]:
    """Return human-readable debug lines for exploration index data."""
    lines: list[str] = []

    for attribute_name, label in (
        ("index", "index"),
        ("zipf_factored_proba", "zipf_factored_proba"),
        ("min_path_value", "min_path_value"),
        ("max_path_value", "max_path_value"),
        ("max_depth_descendants", "max_depth_descendants"),
    ):
        value = safe_getattr(index_data, attribute_name)
        if value is not None:
            lines.append(f"{label}={value}")

    interval = safe_getattr(index_data, "interval")
    if interval is not None:
        min_value = safe_getattr(interval, "min_value")
        max_value = safe_getattr(interval, "max_value")
        if min_value is not None or max_value is not None:
            lines.append(f"interval=[{min_value}, {max_value}]")

    return lines if lines else [f"index={index_data}"]
