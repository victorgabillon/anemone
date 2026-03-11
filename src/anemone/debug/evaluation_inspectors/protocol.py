"""Protocols for evaluation debug inspectors."""

from __future__ import annotations

from typing import Any, Protocol


class EvaluationDebugInspector(Protocol):
    """Describe how an evaluation object should be summarized for debugging."""

    def supports(self, evaluation: Any) -> bool:
        """Return whether this inspector can summarize ``evaluation``."""
        ...

    def summarize(self, evaluation: Any) -> list[str]:
        """Return human-readable label lines for ``evaluation``."""
        ...
