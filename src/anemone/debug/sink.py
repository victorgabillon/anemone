"""Protocols and helpers for consuming search debug events."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from anemone.debug.events import SearchDebugEvent


class SearchDebugSink(Protocol):
    """Receive structured search debug events."""

    def emit(self, event: SearchDebugEvent) -> None:
        """Consume one search debug event."""
        ...


class NullSearchDebugSink:
    """No-op debug sink."""

    def emit(self, event: SearchDebugEvent) -> None:
        """Ignore one debug event."""
        _ = event


__all__ = ["NullSearchDebugSink", "SearchDebugSink"]
