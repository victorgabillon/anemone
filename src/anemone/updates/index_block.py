"""
This module defines the IndexUpdateInstructionsBlock class, which represents a block of update instructions for
index values in a tree structure.

The IndexUpdateInstructionsBlock class is a dataclass that contains a set of AlgorithmNode objects representing
children with updated index values. It provides methods for merging update instructions and printing information
about the block.
"""

from dataclasses import dataclass, field
from typing import Self

from valanga import BranchKey


from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)


@dataclass(slots=True)
class IndexUpdateInstructionsFromOneNode:
    """
    Represents a block of instructions for updating an index.

    Attributes:
        node_sending_update (AlgorithmNode): The node sending the update.
        updated_index (bool): Indicates whether the index has been updated.
    """

    node_sending_update: AlgorithmNode
    updated_index: bool


@dataclass(slots=True)
class IndexUpdateInstructionsTowardsOneParentNode:
    """
    Represents a block of index update instructions intended to a specific node in the algorithm tree.

    This class is used to store and manipulate sets of children with updated index values.

    Attributes:
        moves_with_updated_index (Set[IMove]): A set of children with updated index values.
    """

    branches_with_updated_index: set[BranchKey] = field(
        default_factory=lambda: set[BranchKey]()
    )

    def add_update_from_one_child_node(
        self,
        update_from_one_child_node: IndexUpdateInstructionsFromOneNode,
        branch_from_parent_to_child: BranchKey,
    ) -> None:
        """Adds an update from a child node to the parent node.

        Args:
            update_from_one_child_node (IndexUpdateInstructionsFromOneNode): The update instructions from the child node.
            move_from_parent_to_child (moveKey): The move key representing the parent's move to the child.
        """
        if update_from_one_child_node.updated_index:
            self.branches_with_updated_index.add(branch_from_parent_to_child)

    def add_update_toward_one_parent_node(self, another_update: Self) -> None:
        """Adds an update from another child node to the parent node.

        Args:
            another_update (Self): The update instructions from another child node.
        """
        self.branches_with_updated_index = (
            self.branches_with_updated_index | another_update.branches_with_updated_index
        )

    def empty(self) -> bool:
        """
        Check if the IndexUpdateInstructionsBlock is empty.

        Returns:
            bool: True if the block is empty, False otherwise.
        """
        empty_bool = not bool(self.branches_with_updated_index)
        return empty_bool

    def print_info(self) -> None:
        """Prints information about the moves with updated indices."""
        print(self.branches_with_updated_index)
