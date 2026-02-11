"""Protocol for priority checks used by composed node selectors."""

from typing import TYPE_CHECKING, Any, Protocol

from anemone import trees
from anemone.node_selector.opening_instructions import OpeningInstructions
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

if TYPE_CHECKING:
    from anemone import tree_manager as tree_man


class PriorityCheck[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](Protocol):
    """Priority override hook that can optionally choose an opening."""

    def maybe_choose_opening(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> OpeningInstructions[NodeT] | None:
        """Return override opening instructions, or ``None`` to fall back to base selector."""
        raise NotImplementedError
