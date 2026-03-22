"""Search-time extension hooks for integrator-provided features."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from random import Random
from typing import Any, Protocol

from anemone._valanga_types import AnyTurnState
from anemone.node_selector.opening_instructions import OpeningInstructor
from anemone.node_selector.priority_check.priority_check import PriorityCheck

type PriorityCheckFactory = Callable[
    [Mapping[str, Any], Random, "SearchHooks | None", OpeningInstructor],
    PriorityCheck,
]


class FeatureExtractor(Protocol):
    """Extract arbitrary features from game states for optional selector logic."""

    def features(self, state: AnyTurnState) -> Mapping[str, Any]:
        """Return a mapping of feature names to values for the given state."""
        raise NotImplementedError


def _empty_priority_registry() -> dict[str, PriorityCheckFactory]:
    return {}


@dataclass(frozen=True)
class SearchHooks:
    """Container for optional search extension points."""

    feature_extractor: FeatureExtractor | None = None
    priority_check_registry: Mapping[str, PriorityCheckFactory] = field(
        default_factory=_empty_priority_registry
    )
