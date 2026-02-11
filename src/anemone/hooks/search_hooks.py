"""Search-time extension hooks for integrator-provided features."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from valanga import TurnState


class FeatureExtractor(Protocol):
    """Extract arbitrary features from game states for optional selector logic."""

    def features(self, state: TurnState) -> Mapping[str, Any]:
        """Return a mapping of feature names to values for the given state."""
        raise NotImplementedError


@dataclass(frozen=True)
class SearchHooks:
    """Container for optional search extension points."""

    feature_extractor: FeatureExtractor | None = None
