"""
This module contains classes for managing update instructions in a batch.

Classes:
- UpdateInstructions: Represents update instructions for a single node.
- UpdateInstructionsBatch: Represents a batch of update instructions for multiple nodes.
"""

from dataclasses import dataclass, field
from typing import Self

from valanga import BranchKey

from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.utils.dict_of_numbered_dict_with_pointer_on_max import (
    DictOfNumberedDictWithPointerOnMax,
)

from .index_block import (
    IndexUpdateInstructionsFromOneNode,
    IndexUpdateInstructionsTowardsOneParentNode,
)
from .value_block import (
    ValueUpdateInstructionsFromOneNode,
    ValueUpdateInstructionsTowardsOneParentNode,
)


@dataclass(slots=True)
class UpdateInstructionsFromOneNode:
    """
    Represents update instructions generated from a single node.

    Attributes:
    - value_block: The value update instructions generated from a single node.
    - index_block: The index update instructions generated from a single node.
    """

    value_block: ValueUpdateInstructionsFromOneNode | None = None
    index_block: IndexUpdateInstructionsFromOneNode | None = None


@dataclass(slots=True)
class UpdateInstructionsTowardsOneParentNode:
    """
    Represents update instructions for a single node.

    Attributes:
    - value_block: The value update instructions block.
    - index_block: The index update instructions block.
    """

    value_updates_toward_one_parent_node: (
        ValueUpdateInstructionsTowardsOneParentNode | None
    ) = None
    index_updates_toward_one_parent_node: (
        IndexUpdateInstructionsTowardsOneParentNode | None
    ) = None

    def add_update_from_a_child_node(
        self,
        update_from_a_child_node: UpdateInstructionsFromOneNode,
        branch_from_parent_to_child: BranchKey,
    ) -> None:
        """
        Adds update instructions from a child node.

        Args:
        - update_from_a_child_node: The update instructions from the child node.
        - branch_from_parent_to_child: The branch key from the parent to the child.
        """
        assert self.value_updates_toward_one_parent_node is not None
        assert update_from_a_child_node.value_block is not None
        self.value_updates_toward_one_parent_node.add_update_from_one_child_node(
            branch_from_parent_to_child=branch_from_parent_to_child,
            update_from_one_child_node=update_from_a_child_node.value_block,
        )

        if self.index_updates_toward_one_parent_node is None:
            assert update_from_a_child_node.index_block is None
        else:
            if update_from_a_child_node.index_block is not None:
                self.index_updates_toward_one_parent_node.add_update_from_one_child_node(
                    branch_from_parent_to_child=branch_from_parent_to_child,
                    update_from_one_child_node=update_from_a_child_node.index_block,
                )

    def add_updates_towards_one_parent_node(self, another_update: Self) -> None:
        """
        Adds update instructions from another UpdateInstructionsTowardsOneParentNode.
        Args:
        - another_update: The other update instructions to add.
        """
        assert self.value_updates_toward_one_parent_node is not None
        assert another_update.value_updates_toward_one_parent_node is not None
        self.value_updates_toward_one_parent_node.add_update_toward_one_parent_node(
            another_update.value_updates_toward_one_parent_node
        )

        if self.index_updates_toward_one_parent_node is None:
            assert another_update.index_updates_toward_one_parent_node is None
        else:
            if another_update.index_updates_toward_one_parent_node is not None:
                self.index_updates_toward_one_parent_node.add_update_toward_one_parent_node(
                    another_update.index_updates_toward_one_parent_node
                )

    def print_info(self) -> None:
        """
        Prints information about the update instructions.
        """
        print("printing info of update instructions")
        assert (
            self.index_updates_toward_one_parent_node is not None
            and self.value_updates_toward_one_parent_node is not None
        )
        self.value_updates_toward_one_parent_node.print_info()
        self.index_updates_toward_one_parent_node.print_info()

    def empty(self) -> bool:
        """
        Checks if the update instructions are empty.

        Returns:
        - True if the update instructions are empty, False otherwise.
        """
        assert self.value_updates_toward_one_parent_node is not None
        return self.value_updates_toward_one_parent_node.empty() and (
            self.index_updates_toward_one_parent_node is None
            or self.index_updates_toward_one_parent_node.empty()
        )


@dataclass
class UpdateInstructionsTowardsMultipleNodes[NodeT: AlgorithmNode = AlgorithmNode]:
    """Represents update instructions towards multiple parent nodes."""

    @staticmethod
    def _new_one_node_instructions() -> DictOfNumberedDictWithPointerOnMax[
        NodeT, UpdateInstructionsTowardsOneParentNode
    ]:
        """Return a fresh mapping for per-node update instructions."""
        return DictOfNumberedDictWithPointerOnMax()

    one_node_instructions: DictOfNumberedDictWithPointerOnMax[
        NodeT, UpdateInstructionsTowardsOneParentNode
    ] = field(default_factory=_new_one_node_instructions)

    def add_update_from_one_child_node(
        self,
        update_from_child_node: UpdateInstructionsFromOneNode,
        parent_node: NodeT,
        branch_from_parent: BranchKey,
    ) -> None:
        """
        Adds update instructions from a child node to a parent node.
        Args:
            update_from_child_node: The update instructions from the child node.
            parent_node: The parent node to which the updates are directed.
            branch_from_parent: The branch key from the parent to the child.
        """
        if parent_node not in self.one_node_instructions:
            # build the UpdateInstructionsTowardsOneParentNode
            assert update_from_child_node.value_block is not None
            value_updates_toward_one_parent_node: (
                ValueUpdateInstructionsTowardsOneParentNode
            )
            value_updates_toward_one_parent_node = (
                ValueUpdateInstructionsTowardsOneParentNode(
                    branches_with_updated_value=(
                        {branch_from_parent}
                        if update_from_child_node.value_block.new_value_for_node
                        else set()
                    ),
                    branches_with_updated_over=(
                        {branch_from_parent}
                        if update_from_child_node.value_block.is_node_newly_over
                        else set()
                    ),
                    branches_with_updated_best_branch=(
                        {branch_from_parent}
                        if update_from_child_node.value_block.new_best_move_for_node
                        else set()
                    ),
                )
            )
            index_updates_toward_one_parent_node: (
                IndexUpdateInstructionsTowardsOneParentNode | None
            )
            if update_from_child_node.index_block is not None:
                index_updates_toward_one_parent_node = (
                    IndexUpdateInstructionsTowardsOneParentNode(
                        branches_with_updated_index=(
                            {branch_from_parent}
                            if update_from_child_node.index_block.updated_index
                            else set()
                        ),
                    )
                )
            else:
                index_updates_toward_one_parent_node = None
            update_instructions_towards_parent: UpdateInstructionsTowardsOneParentNode
            update_instructions_towards_parent = UpdateInstructionsTowardsOneParentNode(
                value_updates_toward_one_parent_node=value_updates_toward_one_parent_node,
                index_updates_toward_one_parent_node=index_updates_toward_one_parent_node,
            )
            self.one_node_instructions[parent_node] = update_instructions_towards_parent
        else:
            # update the UpdateInstructionsTowardsOneParentNode
            self.one_node_instructions[parent_node].add_update_from_a_child_node(
                update_from_a_child_node=update_from_child_node,
                branch_from_parent_to_child=branch_from_parent,
            )

    def add_updates_towards_one_parent_node(
        self,
        update_from_child_node: UpdateInstructionsTowardsOneParentNode,
        parent_node: NodeT,
    ) -> None:
        """
        Adds update instructions from another UpdateInstructionsTowardsOneParentNode to a parent node.
        Args:
            update_from_child_node: The update instructions from another UpdateInstructionsTowardsOneParentNode.
            parent_node: The parent node to which the updates are directed.
        """
        if parent_node in self.one_node_instructions:
            self.one_node_instructions[parent_node].add_updates_towards_one_parent_node(
                another_update=update_from_child_node
            )
        else:
            self.one_node_instructions[parent_node] = update_from_child_node

    def pop_item(self) -> tuple[NodeT, UpdateInstructionsTowardsOneParentNode]:
        """
        Pops an item from the update instructions.
        Returns:
            A tuple containing the parent node and its corresponding update instructions.
        """
        return self.one_node_instructions.popitem()

    def __bool__(self) -> bool:
        """
        Checks if the data structure is non-empty.

        Returns:
            bool: True if the data structure is non-empty, False otherwise.
        """
        return bool(self.one_node_instructions)
