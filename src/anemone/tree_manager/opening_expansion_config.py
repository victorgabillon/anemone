"""Configuration objects for opening expansion strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class OpeningExpansionKind(StrEnum):
    """Available opening expansion strategies."""

    ONE_PLY = "one_ply"
    ROLLOUT = "rollout"


class RolloutActionSelectorKind(StrEnum):
    """Built-in rollout continuation action selectors."""

    FIRST_OPENABLE = "first_openable"
    RANDOM_OPENABLE = "random_openable"
    NO_ROLLOUT = "no_rollout"


@dataclass(frozen=True, slots=True)
class RolloutExpansionConfig:
    """Configuration for materialized rollout expansion.

    ``max_extra_steps`` limits continuation decisions after each initial edge.
    A continuation decision may traverse an already-opened edge or materialize
    a new edge at the frontier. If ``None``, rollout continues until a normal
    stop condition such as terminal state, no legal action, action-selector
    stop, branch-budget exhaustion, or existing-node stop.
    """

    max_extra_steps: int | None = 0
    action_selector_kind: RolloutActionSelectorKind = (
        RolloutActionSelectorKind.FIRST_OPENABLE
    )
    stop_on_existing_node: bool = True
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate rollout expansion settings."""
        if self.max_extra_steps is not None and self.max_extra_steps < 0:
            msg = "max_extra_steps must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class OpeningExpansionConfig:
    """Configuration for opening expansion execution."""

    kind: OpeningExpansionKind = OpeningExpansionKind.ONE_PLY
    rollout: RolloutExpansionConfig = field(default_factory=RolloutExpansionConfig)


__all__ = [
    "OpeningExpansionConfig",
    "OpeningExpansionKind",
    "RolloutActionSelectorKind",
    "RolloutExpansionConfig",
]
