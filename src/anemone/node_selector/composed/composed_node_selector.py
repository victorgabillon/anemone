"""Composed node selector with optional priority override and base fallback."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from anemone import trees
from anemone.node_selector.node_selector import (
    InvalidatableNodeSelector,
    NodeSelector,
)
from anemone.node_selector.opening_instructions import OpeningInstructions
from anemone.node_selector.priority_check.priority_check import PriorityCheck
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.objectives import SingleAgentMaxObjective

if TYPE_CHECKING:
    from collections.abc import Callable

    from anemone import tree_manager as tree_man
    from anemone.checkpoints.payloads import SelectorCheckpointPayload


@dataclass(frozen=True)
class ComposedNodeSelector[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](
    NodeSelector[NodeT]
):
    """Node selector composed of a priority check and a base selector."""

    priority_check: PriorityCheck[NodeT]
    base: NodeSelector[NodeT]
    latest_selection_report: object | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> OpeningInstructions[NodeT]:
        """Use priority override when available, otherwise defer to base selector."""
        opening_instructions = self.priority_check.maybe_choose_opening(
            tree, latest_tree_expansions
        )
        if opening_instructions is not None:
            object.__setattr__(self, "latest_selection_report", None)
            return opening_instructions
        opening_instructions = self.base.choose_node_and_branch_to_open(
            tree, latest_tree_expansions
        )
        object.__setattr__(
            self,
            "latest_selection_report",
            getattr(self.base, "latest_selection_report", None),
        )
        return opening_instructions

    def invalidate(self) -> None:
        """Forward selector-cache invalidation to the base selector when supported."""
        if not isinstance(self.base, InvalidatableNodeSelector):
            return
        invalidate = cast(
            "Callable[[], None] | None",
            getattr(self.base, "invalidate", None),
        )
        if invalidate is None:
            return
        invalidate()  # pylint: disable=not-callable

    def refresh_state_for_checkpoint(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> None:
        """Forward selector checkpoint refresh to the base selector when supported."""
        if not isinstance(self.base, InvalidatableNodeSelector):
            return
        refresh = cast(
            "Callable[..., None] | None",
            getattr(self.base, "refresh_state_for_checkpoint", None),
        )
        if refresh is None:
            return
        refresh(  # pylint: disable=not-callable
            tree=tree,
            objective=objective,
            latest_tree_expansions=latest_tree_expansions,
        )

    def build_checkpoint_payload(
        self,
        objective: SingleAgentMaxObjective[Any],
    ) -> "SelectorCheckpointPayload | None":
        """Forward selector checkpoint export to the base selector when supported."""
        if not isinstance(self.base, InvalidatableNodeSelector):
            return None
        build_payload = cast(
            "Callable[[SingleAgentMaxObjective[Any]], SelectorCheckpointPayload | None] | None",
            getattr(self.base, "build_checkpoint_payload", None),
        )
        if build_payload is None:
            return None
        return build_payload(objective)  # pylint: disable=not-callable

    def restore_from_checkpoint_payload(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
        payload: "SelectorCheckpointPayload",
    ) -> bool:
        """Forward selector checkpoint restore to the base selector when supported."""
        if not isinstance(self.base, InvalidatableNodeSelector):
            return False
        restore = cast(
            "Callable[..., bool] | None",
            getattr(self.base, "restore_from_checkpoint_payload", None),
        )
        if restore is None:
            return False
        return restore(  # pylint: disable=not-callable
            tree=tree,
            objective=objective,
            payload=payload,
        )

    def __str__(self) -> str:
        """Return selector composition description."""
        return f"ComposedNodeSelector(base={self.base}, priority={self.priority_check})"
