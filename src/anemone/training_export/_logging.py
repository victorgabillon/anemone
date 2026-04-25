"""Structured logging helpers for training-export instrumentation."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any

from anemone.utils.logger import anemone_logger


def _log_training_export_event(
    phase_name: str,
    status: str,
    **metadata: object,
) -> None:
    """Emit one structured training-export log line."""
    metadata_suffix = " ".join(
        f"{key}={value}" for key, value in metadata.items() if value is not None
    )
    message = f"[training-export] phase={phase_name} status={status}"
    if metadata_suffix:
        message = f"{message} {metadata_suffix}"
    anemone_logger.info(message)


@contextmanager
def _log_training_export_phase(phase_name: str, **metadata: object) -> Any:
    """Log one training-export phase start and completion timing."""
    _log_training_export_event(phase_name, "start", **metadata)
    started_at = perf_counter()
    try:
        yield
    finally:
        _log_training_export_event(
            phase_name,
            "done",
            elapsed_s=round(perf_counter() - started_at, 6),
            **metadata,
        )


__all__ = ["_log_training_export_event", "_log_training_export_phase"]
