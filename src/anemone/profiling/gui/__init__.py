"""Local Streamlit dashboard helpers for Anemone profiling artifacts."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def get_streamlit() -> Any:
    """Return the optional Streamlit module or raise a helpful runtime error."""
    try:
        return import_module("streamlit")
    except ImportError as exc:  # pragma: no cover - exercised through launcher flow
        raise _missing_streamlit_error() from exc


def get_pandas() -> Any | None:
    """Return pandas when available, otherwise ``None`` for graceful fallbacks."""
    try:
        return import_module("pandas")
    except ImportError:  # pragma: no cover - optional dependency fallback
        return None


def _missing_streamlit_error() -> RuntimeError:
    return RuntimeError(
        "Streamlit is not installed. Install the GUI extra with "
        "`pip install -e .[gui]` or install `streamlit` directly."
    )
