"""Composed node selector with optional priority override and base fallback."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anemone import trees
from anemone.node_selector.node_selector import NodeSelector
from anemone.node_selector.opening_instructions import OpeningInstructions
from anemone.node_selector.priority_check.priority_check import PriorityCheck
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

if TYPE_CHECKING:
    from anemone import tree_manager as tree_man


@dataclass(frozen=True)
class ComposedNodeSelector[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](
    NodeSelector[NodeT]
):
    """Node selector composed of a priority check and a base selector."""

    priority_check: PriorityCheck[NodeT]
    base: NodeSelector[NodeT]

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
            return opening_instructions
        return self.base.choose_node_and_branch_to_open(tree, latest_tree_expansions)

    def __str__(self) -> str:
        """Return selector composition description."""
        return f"ComposedNodeSelector(base={self.base}, priority={self.priority_check})"
