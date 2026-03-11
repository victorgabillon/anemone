"""Module that contains the classes for the exploration data of a tree node."""

from dataclasses import dataclass, field
from typing import Any

from valanga import State

from anemone.nodes.itree_node import ITreeNode
from anemone.nodes.tree_node import TreeNode
from anemone.utils.small_tools import Interval


@dataclass
class NodeExplorationData[
    Node: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
]:
    """Represents the exploration data for a tree node.

    Attributes:
        tree_node (TreeNode): The tree node associated with the exploration data.
        index (float | None): The index value associated with the node. Defaults to None.

    """

    tree_node: TreeNode[Node, StateT]
    index: float | None = None


@dataclass
class RecurZipfQuoolExplorationData[
    Node: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
](NodeExplorationData[Node, StateT]):
    """Represents the exploration data for a tree node with recursive zipf-quool factor.

    Attributes:
        zipf_factored_proba (float | None): The probability associated with the node, factored by zipf-quool factor.
            Defaults to None.

    """

    # the 'proba' associated by recursively multiplying 1/rank of the node with the max zipf_factor of the parents
    zipf_factored_proba: float | None = None


@dataclass
class MinMaxPathValue[
    Node: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
](NodeExplorationData[Node, StateT]):
    """Represents the exploration data for a tree node with minimum and maximum path values.

    Attributes:
        min_path_value (float | None): The minimum path value associated with the node. Defaults to None.
        max_path_value (float | None): The maximum path value associated with the node. Defaults to None.

    """

    min_path_value: float | None = None
    max_path_value: float | None = None


@dataclass
class IntervalExplo[
    Node: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
](NodeExplorationData[Node, StateT]):
    """Represents the exploration data for a tree node with an interval.

    Attributes:
        interval (Interval | None): The interval associated with the node. Defaults to None.

    """

    interval: Interval | None = field(default_factory=Interval)


@dataclass
class MaxDepthDescendants[
    Node: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
](NodeExplorationData[Node, StateT]):
    """Represents the exploration data for a tree node with maximum depth of descendants."""

    max_depth_descendants: int = 0

    def update_from_child(self, child_max_depth_descendants: int) -> bool:
        """Update the max_depth_descendants value based on the child's max_depth_descendants.

        Args:
            child_max_depth_descendants (int): The max_depth_descendants value of the child node.

        Returns:
            bool: True if the max_depth_descendants value has changed, False otherwise.

        """
        previous_index = self.max_depth_descendants
        new_index: int = max(
            self.max_depth_descendants, child_max_depth_descendants + 1
        )
        self.max_depth_descendants = new_index
        has_index_changed: bool = new_index != previous_index

        return has_index_changed
