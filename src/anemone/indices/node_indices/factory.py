"""This module provides functions for creating exploration index data for tree nodes.

The main function in this module is `create_exploration_index_data`, which takes a tree node and optional parameters
to create the exploration index data for that node.

The module also defines the `ExplorationIndexDataFactory` type, which is a callable type for creating exploration index data.

Functions:
- create_exploration_index_data: Creates exploration index data for a given tree node.

Types:
- ExplorationIndexDataFactory: A callable type for creating exploration index data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from valanga import State

from anemone.indices.node_indices.index_data import (
    IntervalExplo,
    MaxDepthDescendants,
    MinMaxPathValue,
    NodeExplorationData,
    RecurZipfQuoolExplorationData,
)
from anemone.indices.node_indices.index_types import (
    IndexComputationType,
)
from anemone.nodes.itree_node import ITreeNode
from anemone.nodes.tree_node import TreeNode


@dataclass
class DepthExtendedIntervalExplo[
    T: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
](IntervalExplo[T, StateT], MaxDepthDescendants[T, StateT]):
    """Depth extended interval exploration data."""


@dataclass
class DepthExtendedMinMaxPathValue[
    T: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
](MinMaxPathValue[T, StateT], MaxDepthDescendants[T, StateT]):
    """Depth extended min-max path value exploration data."""


@dataclass
class DepthExtendedRecurZipfQuoolExplorationData[
    T: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
](RecurZipfQuoolExplorationData[T, StateT], MaxDepthDescendants[T, StateT]):
    """Depth extended recur zipf quool exploration data."""


# Generic factory type that preserves the node type through the TreeNode parameter
type ExplorationIndexDataFactory[
    T: ITreeNode[Any] = ITreeNode[Any],
    StateT: State = State,
] = Callable[[TreeNode[T, StateT]], NodeExplorationData[T, StateT] | None]


def create_exploration_index_data[
    NodeT: ITreeNode[Any] = ITreeNode[Any],
    NodeStateT: State = State,
](
    tree_node: TreeNode[NodeT, NodeStateT],
    index_computation: IndexComputationType | None = None,
    depth_index: bool = False,
) -> NodeExplorationData[NodeT, NodeStateT] | None:
    """Creates exploration index data for a given tree node.

    Args:
        tree_node (TreeNode): The tree node for which to create the exploration index data.
        index_computation (IndexComputationType | None, optional): The type of index computation to use. Defaults to None.
        depth_index (bool, optional): Whether to include depth information in the index data. Defaults to False.

    Returns:
        NodeExplorationData | None: The created exploration index data.

    Raises:
        ValueError: If the index_computation value is not recognized.

    """
    index_dataclass_name: type[Any] | None
    match index_computation:
        case None:
            index_dataclass_name = None
        case IndexComputationType.MIN_LOCAL_CHANGE:
            index_dataclass_name = (
                DepthExtendedIntervalExplo if depth_index else IntervalExplo
            )
        case IndexComputationType.MIN_GLOBAL_CHANGE:
            index_dataclass_name = (
                DepthExtendedMinMaxPathValue if depth_index else MinMaxPathValue
            )
        case IndexComputationType.RECUR_ZIPF:
            index_dataclass_name = (
                DepthExtendedRecurZipfQuoolExplorationData
                if depth_index
                else RecurZipfQuoolExplorationData
            )
        case _:
            raise ValueError(
                f"not finding good case for {index_computation} in file {__name__}"
            )

    exploration_index_data: NodeExplorationData[NodeT, NodeStateT] | None
    exploration_index_data = (
        index_dataclass_name(tree_node=tree_node) if index_dataclass_name else None
    )
    return exploration_index_data
