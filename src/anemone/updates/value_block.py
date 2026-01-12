"""
This module defines the ValueUpdateInstructionsBlock class and a helper function to create instances of it.

The ValueUpdateInstructionsBlock class represents a block of update instructions for a tree value node in
 a branch selector algorithm. It contains sets of branches that have been updated with new values,
  best branches, or are newly over.

The create_value_update_instructions_block function is a helper function that creates an instance of
 the ValueUpdateInstructionsBlock class with the specified update instructions.

"""

from dataclasses import dataclass, field
from typing import Self

from valanga import BranchKey

from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)


@dataclass(slots=True)
class ValueUpdateInstructionsFromOneNode:
    """Represents update instructions generated from a single node."""

    node_sending_update: AlgorithmNode
    is_node_newly_over: bool
    new_value_for_node: bool
    new_best_branch_for_node: bool


def _new_branchkey_set() -> set[BranchKey]:
    """Return a new empty set of branch keys."""
    return set()


@dataclass(slots=True)
class ValueUpdateInstructionsTowardsOneParentNode:
    """Represents a block of value-update instructions intended to a specific node in the algorithm tree."""

    branches_with_updated_over: set[BranchKey] = field(
        default_factory=_new_branchkey_set
    )
    branches_with_updated_value: set[BranchKey] = field(
        default_factory=_new_branchkey_set
    )
    branches_with_updated_best_branch: set[BranchKey] = field(
        default_factory=_new_branchkey_set
    )

    def add_update_from_one_child_node(
        self,
        update_from_one_child_node: ValueUpdateInstructionsFromOneNode,
        branch_from_parent_to_child: BranchKey,
    ) -> None:
        """Adds an update from a child node to the parent node.

        Args:
            update_from_one_child_node (ValueUpdateInstructionsFromOneNode): The update instructions from the child node.
            move_from_parent_to_child (moveKey): The branch key representing the branch from the parent to the child.
        """
        if update_from_one_child_node.is_node_newly_over:
            self.branches_with_updated_over.add(branch_from_parent_to_child)
        if update_from_one_child_node.new_value_for_node:
            self.branches_with_updated_value.add(branch_from_parent_to_child)
        if update_from_one_child_node.new_best_branch_for_node:
            self.branches_with_updated_best_branch.add(branch_from_parent_to_child)

    def add_update_toward_one_parent_node(self, another_update: Self) -> None:
        """Adds an update towards one parent node.

        Args:
            another_update (Self): The update instructions from another child node.
        """
        self.branches_with_updated_value = (
            self.branches_with_updated_value
            | another_update.branches_with_updated_value
        )
        self.branches_with_updated_over = (
            self.branches_with_updated_over | another_update.branches_with_updated_over
        )
        self.branches_with_updated_best_branch = (
            self.branches_with_updated_best_branch
            | another_update.branches_with_updated_best_branch
        )

    def print_info(self) -> None:
        """
        Print information about the update instructions block.

        Returns:
            None
        """
        print("upInstructions printing")
        print(
            len(self.branches_with_updated_value),
            "branches_with_updated_value",
            end=" ",
        )
        for branch in self.branches_with_updated_value:
            print(branch, end=" ")
        print(
            "\n",
            len(self.branches_with_updated_best_branch),
            "branches_with_updated_best_branch:",
            end=" ",
        )
        for branch in self.branches_with_updated_best_branch:
            print(branch, end=" ")
        print(
            "\n",
            len(self.branches_with_updated_over),
            "branches_with_updated_over",
            end=" ",
        )
        for branch in self.branches_with_updated_over:
            print(branch, end=" ")
        print()

    def empty(self) -> bool:
        """
        Check if all the components of the update instructions block are empty.

        Returns:
            bool: True if all components are empty, False otherwise.
        """
        empty_bool = (
            not bool(self.branches_with_updated_value)
            and not bool(self.branches_with_updated_best_branch)
            and not bool(self.branches_with_updated_over)
        )
        return empty_bool
