"""Shared helpers for GUI chart rendering."""

from __future__ import annotations

from typing import Any

from anemone.profiling.gui import get_altair, get_pandas, get_streamlit


def get_chart_modules() -> tuple[Any, Any, Any]:
    """Return Streamlit, pandas, and Altair modules for chart rendering."""
    st: Any = get_streamlit()
    pandas_module = get_pandas()
    altair_module = get_altair()
    return st, pandas_module, altair_module


def build_chart_dataframe(
    rows: object,
    *,
    fallback_rows: object | None = None,
) -> tuple[Any, Any, Any] | None:
    """Return Streamlit, Altair, and a DataFrame or render a table fallback."""
    st, pandas_module, altair_module = get_chart_modules()
    if pandas_module is None or altair_module is None:
        render_table_fallback(rows if fallback_rows is None else fallback_rows)
        return None
    return st, altair_module, pandas_module.DataFrame(rows)


def build_chart(
    rows: object,
    *,
    fallback_rows: object | None = None,
) -> tuple[Any, Any, Any] | None:
    """Return Streamlit and an Altair chart base or render a table fallback."""
    chart_context = build_chart_dataframe(rows, fallback_rows=fallback_rows)
    if chart_context is None:
        return None
    st, altair_module, dataframe = chart_context
    return st, altair_module, altair_module.Chart(dataframe)


def render_table_fallback(rows: object) -> None:
    """Render a simple dataframe fallback when chart dependencies are unavailable."""
    st: Any = get_streamlit()
    st.dataframe(rows, width="stretch", hide_index=True)
