"""Resolver for evaluation debug inspectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .generic_value_family import GenericValueFamilyInspector
from .minmax_family import MinmaxFamilyInspector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .protocol import EvaluationDebugInspector


class EvaluationDebugInspectorResolver:
    """Resolve the best debug inspector for an evaluation object."""

    def __init__(
        self,
        inspectors: Iterable[EvaluationDebugInspector] | None = None,
    ) -> None:
        """Initialize the resolver with an ordered inspector list."""
        self._inspectors = tuple(
            inspectors
            if inspectors is not None
            else (
                GenericValueFamilyInspector(),
                MinmaxFamilyInspector(),
            )
        )

    def summarize(self, evaluation: Any) -> list[str]:
        """Return the first matching inspector summary for ``evaluation``."""
        for inspector in self._inspectors:
            if inspector.supports(evaluation):
                return inspector.summarize(evaluation)
        return []
