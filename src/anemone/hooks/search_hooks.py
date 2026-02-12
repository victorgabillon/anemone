"""Search-time extension hooks for integrator-provided features."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING, Any, Protocol

from anemone.node_selector.opening_instructions import OpeningInstructor
from anemone.node_selector.priority_check.priority_check import PriorityCheck

if TYPE_CHECKING:
    from valanga import TurnState


type PriorityCheckFactory = Callable[
    [Mapping[str, Any], Random, "SearchHooks | None", OpeningInstructor],
    PriorityCheck,
]


class FeatureExtractor(Protocol):
    """Extract arbitrary features from game states for optional selector logic."""

    def features(self, state: "TurnState") -> Mapping[str, Any]:
        """Return a mapping of feature names to values for the given state."""
        raise NotImplementedError


@dataclass(frozen=True)
class SearchHooks:
    """Container for optional search extension points."""

    feature_extractor: FeatureExtractor | None = None
    priority_check_registry: Mapping[str, PriorityCheckFactory] = field(
        default_factory=dict
    )
