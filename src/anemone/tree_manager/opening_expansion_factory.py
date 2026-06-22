"""Factories for opening expansion executors and rollout action selectors."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING, Any

from anemone import nodes as node
from anemone.rollouts import (
    FirstLegalPreferOpenableActionSelector,
    FirstOpenableActionSelector,
    NoRolloutActionSelector,
    RandomLegalPreferOpenableActionSelector,
    RandomOpenableActionSelector,
    RolloutActionSelector,
    RolloutOpeningExpansionExecutor,
)
from anemone.tree_manager.branch_opening_service import BranchOpeningService
from anemone.tree_manager.opening_expansion_executor import (
    OnePlyOpeningExpansionExecutor,
    OpeningExpansionExecutor,
)

from .opening_expansion_config import (
    OpeningExpansionConfig,
    OpeningExpansionKind,
    RolloutActionSelectorKind,
    RolloutExpansionConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from valanga import BranchKey

    from anemone.dynamics import SearchDynamics

    from .tree_manager import TreeManager


def create_rollout_action_selector[
    NodeT: node.ITreeNode[Any] = node.ITreeNode[Any],
](
    config: RolloutExpansionConfig,
) -> RolloutActionSelector[NodeT]:
    """Create a rollout action selector from configuration."""

    def create_first_openable() -> RolloutActionSelector[NodeT]:
        """Create a first-openable rollout action selector."""
        return FirstOpenableActionSelector[NodeT]()

    def create_first_legal_prefer_openable() -> RolloutActionSelector[NodeT]:
        """Create a first-legal-prefer-openable rollout action selector."""
        return FirstLegalPreferOpenableActionSelector[NodeT]()

    def create_no_rollout() -> RolloutActionSelector[NodeT]:
        """Create a no-rollout action selector."""
        return NoRolloutActionSelector[NodeT]()

    def create_random_openable() -> RolloutActionSelector[NodeT]:
        """Create a random-openable rollout action selector."""
        return RandomOpenableActionSelector[NodeT](Random(config.random_seed))

    def create_random_legal_prefer_openable() -> RolloutActionSelector[NodeT]:
        """Create a random-legal-prefer-openable rollout action selector."""
        return RandomLegalPreferOpenableActionSelector[NodeT](
            Random(config.random_seed)
        )

    selector_factories: dict[
        RolloutActionSelectorKind, Callable[[], RolloutActionSelector[NodeT]]
    ] = {
        RolloutActionSelectorKind.FIRST_OPENABLE: create_first_openable,
        RolloutActionSelectorKind.FIRST_LEGAL_PREFER_OPENABLE: (
            create_first_legal_prefer_openable
        ),
        RolloutActionSelectorKind.NO_ROLLOUT: create_no_rollout,
        RolloutActionSelectorKind.RANDOM_LEGAL_PREFER_OPENABLE: (
            create_random_legal_prefer_openable
        ),
        RolloutActionSelectorKind.RANDOM_OPENABLE: create_random_openable,
    }
    try:
        return selector_factories[config.action_selector_kind]()
    except KeyError as exc:
        msg = (
            f"unsupported rollout action selector kind: {config.action_selector_kind!r}"
        )
        raise ValueError(msg) from exc


def create_opening_expansion_executor[
    NodeT: node.ITreeNode[Any] = node.ITreeNode[Any],
](
    *,
    config: OpeningExpansionConfig,
    tree_manager: TreeManager[NodeT],
    dynamics: SearchDynamics[Any, Any],
    on_branch_opened: Callable[[NodeT, BranchKey], None] | None = None,
    rollout_action_selector: RolloutActionSelector[NodeT] | None = None,
) -> OpeningExpansionExecutor[NodeT]:
    """Create the configured opening expansion executor.

    ``rollout_action_selector`` is a runtime object override used only for
    rollout expansion. When omitted, rollout config creates the built-in
    selector.
    """
    branch_opening_service = BranchOpeningService(
        tree_manager=tree_manager,
        on_branch_opened=on_branch_opened,
    )

    def create_one_ply() -> OpeningExpansionExecutor[NodeT]:
        """Create a one-ply opening expansion executor."""
        return OnePlyOpeningExpansionExecutor(
            branch_opening_service=branch_opening_service,
            dynamics=dynamics,
        )

    def create_rollout() -> OpeningExpansionExecutor[NodeT]:
        """Create a rollout opening expansion executor."""
        rollout_config = config.rollout
        selected_rollout_action_selector: RolloutActionSelector[NodeT] = (
            rollout_action_selector
            if rollout_action_selector is not None
            else create_rollout_action_selector(rollout_config)
        )
        return RolloutOpeningExpansionExecutor[NodeT](
            branch_opening_service=branch_opening_service,
            dynamics=dynamics,
            rollout_action_selector=selected_rollout_action_selector,
            max_extra_steps=rollout_config.max_extra_steps,
            stop_on_existing_node=rollout_config.stop_on_existing_node,
        )

    executor_factories: dict[
        OpeningExpansionKind, Callable[[], OpeningExpansionExecutor[NodeT]]
    ] = {
        OpeningExpansionKind.ONE_PLY: create_one_ply,
        OpeningExpansionKind.ROLLOUT: create_rollout,
    }
    try:
        return executor_factories[config.kind]()
    except KeyError as exc:
        msg = f"unsupported opening expansion kind: {config.kind!r}"
        raise ValueError(msg) from exc


__all__ = [
    "create_opening_expansion_executor",
    "create_rollout_action_selector",
]
