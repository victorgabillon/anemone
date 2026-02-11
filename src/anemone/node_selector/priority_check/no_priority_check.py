"""No-op priority check implementation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anemone import trees
from anemone.node_selector.opening_instructions import OpeningInstructions
from anemone.node_selector.priority_check.priority_check import PriorityCheck
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

if TYPE_CHECKING:
    from anemone import tree_manager as tree_man


@dataclass(frozen=True)
class NoPriorityCheck[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](
    PriorityCheck[NodeT]
):
    """Priority check that never overrides base selection."""

    # pylint: disable=useless-return
    def maybe_choose_opening(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> OpeningInstructions[NodeT] | None:
        """Never override; always defer to the base selector."""
        _ = tree
        _ = latest_tree_expansions
        return None

    # pylint: enable=useless-return
