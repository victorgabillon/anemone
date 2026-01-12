"""
This module contains classes and functions related to opening instructions in a chess game.
"""

from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Any, ItemsView, Iterator, Self, ValuesView

from valanga import BranchKey

from anemone import nodes
from anemone.nodes.utils import (
    a_branch_str_sequence_from_root,
    a_branch_key_sequence_from_root,
)

type OpeningInstructionKey = tuple[int, BranchKey]


@dataclass(slots=True)
class OpeningInstruction[NodeT: nodes.ITreeNode[Any] = nodes.ITreeNode[Any]]:
    """
    Represents an opening instruction for a specific node in the game tree.
    """

    node_to_open: NodeT
    branch: BranchKey

    def print_info(self) -> None:
        """
        Prints information about the opening instruction.
        """
        print(
            f"OpeningInstruction: node_to_open {self.node_to_open.id} at hm {self.node_to_open.tree_depth} {self.node_to_open.state}| "
            f"a path from root to node_to_open is {a_branch_key_sequence_from_root(self.node_to_open)} {a_branch_str_sequence_from_root(self.node_to_open)}| "
            f"self.branch {self.branch} {self.node_to_open.state.branch_name_from_key(self.branch)}"
        )


class OpeningInstructions[NodeT: nodes.ITreeNode[Any] = nodes.ITreeNode[Any]]:
    # todo do we need a dict? why not a set? verify

    """
    Represents a collection of opening instructions.
    """

    batch: dict[OpeningInstructionKey, OpeningInstruction[NodeT]]

    def __init__(
        self,
        dictionary: dict[OpeningInstructionKey, OpeningInstruction[NodeT]]
        | None = None,
    ) -> None:
        """
        Initializes the OpeningInstructions object.

        Args:
            dictionary: A dictionary of opening instructions (optional).
        """
        # here i use a dictionary because they are insertion ordered until there is an ordered set in python
        # order is important because some method give a batch where the last element in the batch are prioritary
        self.batch = {}

        if dictionary is not None:
            for key in dictionary:
                self[key] = dictionary[key]

    def __setitem__(
        self, key: OpeningInstructionKey, value: OpeningInstruction[NodeT]
    ) -> None:
        """
        Sets an opening instruction in the collection.

        Args:
            key: The key for the opening instruction.
            value: The opening instruction.
        """
        # key is supposed to be a tuple with (node_to_open,  move_to_play)
        self.batch[key] = value

    def __getitem__(self, key: OpeningInstructionKey) -> OpeningInstruction[NodeT]:
        """
        Retrieves an opening instruction from the collection.

        Args:
            key: The key for the opening instruction.

        Returns:
            The opening instruction.
        """
        # assert(0==1)
        return self.batch[key]

    def __iter__(self) -> Iterator[OpeningInstructionKey]:
        """
        Returns an iterator over the keys of the opening instructions.

        Returns:
            An iterator over the keys.
        """
        return iter(self.batch)

    def __bool__(self) -> bool:
        """
        Checks if the collection is non-empty.

        Returns:
            True if the collection is non-empty, False otherwise.
        """
        return bool(self.batch)

    def merge(self, another_opening_instructions_batch: Self) -> None:
        """
        Merges another batch of opening instructions into the current collection.

        Args:
            another_opening_instructions_batch: Another OpeningInstructions object.
        """
        for (
            opening_instruction_key,
            opening_instruction,
        ) in another_opening_instructions_batch.items():
            if opening_instruction_key not in self.batch:
                self.batch[opening_instruction_key] = opening_instruction

    def pop_items(self, how_many: int, popped: Self) -> None:
        """
        Pops a specified number of opening instructions from the collection.

        Args:
            how_many: The number of opening instructions to pop.
            popped: An OpeningInstructions object to store the popped instructions.
        """
        how_many = min(how_many, len(self.batch))
        for _ in range(how_many):
            key, value = self.batch.popitem()
            popped[key] = value

    def values(self) -> ValuesView[OpeningInstruction[NodeT]]:
        """
        Returns a view of the values in the collection.

        Returns:
            A view of the values.
        """
        return self.batch.values()

    def items(self) -> ItemsView[OpeningInstructionKey, OpeningInstruction[NodeT]]:
        """
        Returns a view of the items (key-value pairs) in the collection.

        Returns:
            A view of the items.
        """
        return self.batch.items()

    def print_info(self) -> None:
        """
        Prints information about the opening instructions in the collection.
        """
        print("OpeningInstructionsBatch: batch contains", len(self.batch), "elements:")
        for _key, opening_instructions in self.batch.items():
            opening_instructions.print_info()

    def __len__(self) -> int:
        """
        Returns the number of opening instructions in the collection.

        Returns:
            The number of opening instructions.
        """
        return len(self.batch)


def create_instructions_to_open_all_branches[NodeT: nodes.ITreeNode[Any]](
    branches_to_play: list[BranchKey], node_to_open: NodeT
) -> OpeningInstructions[NodeT]:
    """
    Creates opening instructions for all possible moves to play from a given node.

    Args:
        moves_to_play: A list of chess moves.
        node_to_open: The node to open.

    Returns:
        An OpeningInstructions object containing the opening instructions.
    """
    opening_instructions_batch: OpeningInstructions[NodeT] = OpeningInstructions()

    for branch_to_play in branches_to_play:
        # at the moment it looks redundant keys are almost the same as values but its clean
        # the keys are here for fast and redundant proof insertion
        # and the values are here for clean data processing
        opening_instructions_batch[(node_to_open.id, branch_to_play)] = (
            OpeningInstruction(node_to_open=node_to_open, branch=branch_to_play)
        )
    #  node_to_open.non_opened_legal_moves.add(move_to_play)
    return opening_instructions_batch


class OpeningType(Enum):
    """
    Represents the type of opening to use.
    """

    ALL_CHILDREN = "all_children"


class OpeningInstructor:
    """
    Represents an opening instructor that provides opening instructions based on a specific opening type.
    """

    def __init__(self, opening_type: OpeningType, random_generator: Random) -> None:
        """
        Initializes the OpeningInstructor object.

        Args:
            opening_type: The type of opening to use.
            random_generator: A random number generator.
        """
        self.opening_type = opening_type
        self.random_generator = random_generator

    def all_branches_to_open(
        self, node_to_open: nodes.ITreeNode[Any]
    ) -> list[BranchKey]:
        """
        Returns a list of all possible branches to open from a given node.

        Args:
            node_to_open: The node to open.

        Returns:
            A list of chess moves.
        """
        if self.opening_type == OpeningType.ALL_CHILDREN:
            node_to_open.all_branches_generated = True
            branches_to_play: list[BranchKey] = node_to_open.state.branch_keys.get_all()

            # this shuffling add randomness to the playing style
            # (it stills depends on the random seed, but if random seed varies then the behavior will be more random)
            # DEACTIVATED ATM BECAUSE I DO NOT UNDERSTAND or FORGOT THE USE CASE: MAYBE DEAD SINCE SEED SYSTEM CHANGED
            # self.random_generator.shuffle(moves_to_play)

        else:
            raise NotImplementedError("Hello-la")
        return branches_to_play
