"""Deterministic rollout action selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from anemone import nodes as node

if TYPE_CHECKING:
    from random import Random

    from valanga import BranchKey


@dataclass(frozen=True, slots=True)
class RolloutDecisionContext[
    NodeT: node.ITreeNode[Any] = node.ITreeNode[Any],
]:
    """Context passed to a rollout action selector for one continuation decision."""

    current_node: NodeT
    openable_actions: tuple[BranchKey, ...]
    rollout_step_index: int


class RolloutActionSelector[
    NodeT: node.ITreeNode[Any] = node.ITreeNode[Any],
](Protocol):
    """Choose one openable rollout action, or stop."""

    def choose_action(
        self,
        context: RolloutDecisionContext[NodeT],
    ) -> BranchKey | None:
        """Return the next rollout action, or ``None`` to stop."""
        ...


@dataclass(frozen=True, slots=True)
class NoRolloutActionSelector[NodeT: node.ITreeNode[Any] = node.ITreeNode[Any]]:
    """Rollout action selector that always stops."""

    def choose_action(  # pylint: disable=useless-return
        self,
        context: RolloutDecisionContext[NodeT],
    ) -> BranchKey | None:
        """Return no action."""
        del context
        return None


@dataclass(frozen=True, slots=True)
class FirstOpenableActionSelector[
    NodeT: node.ITreeNode[Any] = node.ITreeNode[Any],
]:
    """Deterministically choose the first currently openable action."""

    def choose_action(
        self,
        context: RolloutDecisionContext[NodeT],
    ) -> BranchKey | None:
        """Return the first openable action, if one exists."""
        if not context.openable_actions:
            return None
        return context.openable_actions[0]


@dataclass(slots=True)
class RandomOpenableActionSelector[
    NodeT: node.ITreeNode[Any] = node.ITreeNode[Any],
]:
    """Randomly choose one currently openable action using an injected RNG."""

    random_generator: Random

    def choose_action(
        self,
        context: RolloutDecisionContext[NodeT],
    ) -> BranchKey | None:
        """Return a random openable action, if one exists."""
        if not context.openable_actions:
            return None
        return self.random_generator.choice(context.openable_actions)


__all__ = [
    "FirstOpenableActionSelector",
    "NoRolloutActionSelector",
    "RandomOpenableActionSelector",
    "RolloutActionSelector",
    "RolloutDecisionContext",
]
