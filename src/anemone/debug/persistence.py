"""Persistence helpers for debug traces."""

from __future__ import annotations

import pickle
from pathlib import Path

from .recording import DebugTrace


class InvalidDebugTraceError(TypeError):
    """Loaded persistence payload is not a debug trace."""

    def __init__(self, loaded_object: object) -> None:
        """Initialize the error with the invalid loaded object type."""
        super().__init__(f"Loaded object is not a DebugTrace: {type(loaded_object)!r}")


def save_debug_trace(trace: DebugTrace, path: str | Path) -> None:
    """Serialize ``trace`` to ``path`` using pickle."""
    target_path = Path(path)
    with target_path.open("wb") as output_file:
        pickle.dump(trace, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_debug_trace(path: str | Path) -> DebugTrace:
    """Load and validate a ``DebugTrace`` from ``path``."""
    target_path = Path(path)
    with target_path.open("rb") as input_file:
        loaded_trace = pickle.load(input_file)

    if not isinstance(loaded_trace, DebugTrace):
        raise InvalidDebugTraceError(loaded_trace)

    return loaded_trace


__all__ = ["InvalidDebugTraceError", "load_debug_trace", "save_debug_trace"]
