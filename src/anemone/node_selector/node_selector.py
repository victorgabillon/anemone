"""Define the NodeSelector class and related types."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from anemone import trees
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.objectives import SingleAgentMaxObjective

from .opening_instructions import OpeningInstructions

if TYPE_CHECKING:
    from anemone import tree_manager as tree_man
    from anemone.checkpoints.payloads import SelectorCheckpointPayload


@dataclass
class NodeSelectorState:
    """Node Selector State."""


class NodeSelector[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](Protocol):
    """Protocol for Node Selectors."""

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> OpeningInstructions[NodeT]:
        """Select a node from the given tree and return the instructions to open a branch.

        Args:
            tree: The tree containing the nodes.
            latest_tree_expansions: The latest expansions of the tree.

        Returns:
            OpeningInstructions: The instructions to open a branch.

        """
        raise NotImplementedError


@runtime_checkable
class InvalidatableNodeSelector(Protocol):
    """Optional selector interface for runtime cache invalidation."""

    def invalidate(self) -> None:
        """Discard selector runtime cache."""
        raise NotImplementedError


@runtime_checkable
class StatefulNodeSelector[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]](
    InvalidatableNodeSelector,
    Protocol,
):
    """Optional selector interface for runtime state lifecycle."""

    def refresh_state_for_checkpoint(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
        latest_tree_expansions: "tree_man.TreeExpansions[NodeT]",
    ) -> None:
        """Refresh initialized state before checkpoint serialization."""
        raise NotImplementedError

    def build_checkpoint_payload(
        self,
        objective: SingleAgentMaxObjective[Any],
    ) -> "SelectorCheckpointPayload | None":
        """Return optional selector checkpoint state, or ``None``."""
        raise NotImplementedError

    def restore_from_checkpoint_payload(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
        payload: "SelectorCheckpointPayload",
    ) -> bool:
        """Restore selector checkpoint state. Return ``False`` if ignored."""
        raise NotImplementedError
