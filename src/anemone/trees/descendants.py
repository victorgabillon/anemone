"""
This module defines the Descendants and RangedDescendants classes.

Descendants:
- Represents a collection of descendants of a tree node at different half moves.
- Provides methods to add, remove, and access descendants.
- Keeps track of the number of descendants and the number of descendants at each half move.

RangedDescendants:
- Inherits from Descendants and adds the ability to track a range of half moves.
- Provides methods to check if a half move is within the current range or acceptable range.
- Allows adding and removing descendants within the range.
- Provides a method to get the range of half moves.

Note: The Descendants and RangedDescendants classes are used in the chipiron project for move selection in a game.
"""

import typing
from typing import Any, Iterator

from sortedcollections import ValueSortedDict
from valanga import StateTag

from anemone.basics import TreeDepth
from anemone.nodes import ITreeNode


class Descendants[TNode: ITreeNode[Any]]:
    """
    Represents a collection of descendants for a specific half move in a tree.

    Attributes:
        descendants_at_tree_depth (dict[TreeDepth, dict[str, TNode]]): A dictionary that maps a half move to a dictionary of descendants.
        number_of_descendants (int): The total number of descendants in the collection.
        number_of_descendants_at_tree_depth (dict[TreeDepth, int]): A dictionary that maps a half move to the number of descendants at that half move.
        min_tree_depth (int | None): The minimum half move in the collection, or None if the collection is empty.
        max_tree_depth (int | None): The maximum half move in the collection, or None if the collection is empty.
    """

    descendants_at_tree_depth: dict[TreeDepth, dict[StateTag, TNode]]
    number_of_descendants: int
    number_of_descendants_at_tree_depth: dict[TreeDepth, int]
    min_tree_depth: TreeDepth | None
    max_tree_depth: TreeDepth | None

    def __init__(self) -> None:
        """
        Initializes a Descendants object.

        This method initializes the Descendants object by setting up the necessary attributes.

        Attributes:
        - descendants_at_tree_depth (dict): A dictionary to store the descendants at each half move.
        - number_of_descendants (int): The total number of descendants.
        - number_of_descendants_at_tree_depth (dict): A dictionary to store the number of descendants at each half move.
        - min_tree_depth (int or None): The minimum half move.
        - max_tree_depth (int or None): The maximum half move.
        """

        self.descendants_at_tree_depth = {}
        self.number_of_descendants = 0
        self.number_of_descendants_at_tree_depth = {}
        self.min_tree_depth = None
        self.max_tree_depth = None

    def iter_on_all_nodes(
        self,
    ) -> Iterator[tuple[TreeDepth, StateTag, TNode]]:
        return (
            (hm, node_tag, node)
            for hm, nodes_at_hm in self.descendants_at_tree_depth.items()
            for node_tag, node in nodes_at_hm.items()
        )

    def keys(self) -> typing.KeysView[TreeDepth]:
        """
        Returns a view of the keys in the descendants_at_tree_depth dictionary.

        Returns:
            typing.KeysView[TreeDepth]: A view of the keys in the descendants_at_tree_depth dictionary.
        """
        return self.descendants_at_tree_depth.keys()

    def __setitem__(self, tree_depth: TreeDepth, value: dict[StateTag, TNode]) -> None:
        """
        Sets the descendants at a specific half move.

        Args:
            tree_depth (TreeDepth): The half move at which to set the descendants.
            value (dict[str, TNode]): The descendants to set.

        Returns:
            None
        """
        self.descendants_at_tree_depth[tree_depth] = value

    def __getitem__(self, tree_depth: TreeDepth) -> dict[StateTag, TNode]:
        """
        Retrieve the descendants at a specific half move.

        Args:
            tree_depth (TreeDepth): The half move to retrieve the descendants for.

        Returns:
            dict[str, TNode]: A dictionary of descendants at the specified half move.
        """
        return self.descendants_at_tree_depth[tree_depth]

    def __iter__(self) -> typing.Iterator[TreeDepth]:
        """
        Returns an iterator over the descendants at each half move.

        Returns:
            An iterator over the descendants at each half move.
        """
        return iter(self.descendants_at_tree_depth)

    def get_count(self) -> int:
        """
        Returns the number of descendants for the current node.

        Returns:
            int: The number of descendants.
        """
        return self.number_of_descendants

    def contains_node(self, node: TNode) -> bool:
        """
        Checks if the descendants contain a specific node.

        Args:
            node (TNode): The node to check for.

        Returns:
            bool: True if the descendants contain the node, False otherwise.
        """
        if (
            node.tree_depth in self.descendants_at_tree_depth
            and node.tag in self[node.tree_depth]
        ):
            return True
        else:
            return False

    def remove_descendant(self, node: TNode) -> None:
        """
        Removes a descendant node from the tree.

        Args:
            node (TNode): The node to be removed.

        Returns:
            None
        """
        tree_depth = node.tree_depth
        fen = node.tag

        self.number_of_descendants -= 1
        self[tree_depth].pop(fen)
        self.number_of_descendants_at_tree_depth[tree_depth] -= 1
        if self.number_of_descendants_at_tree_depth[tree_depth] == 0:
            self.number_of_descendants_at_tree_depth.pop(tree_depth)
            self.descendants_at_tree_depth.pop(tree_depth)

    def empty(self) -> bool:
        """
        Check if the descendants are empty.

        Returns:
            bool: True if the number of descendants is 0, False otherwise.
        """
        return self.number_of_descendants == 0

    def add_descendant(self, node: TNode) -> None:
        """
        Adds a descendant node to the tree.

        Args:
            node (TNode): The descendant node to be added.

        Returns:
            None
        """
        tree_depth: TreeDepth = node.tree_depth
        node_tag: StateTag = node.tag

        if tree_depth in self.descendants_at_tree_depth:
            assert node_tag not in self.descendants_at_tree_depth[tree_depth]
            self.descendants_at_tree_depth[tree_depth][node_tag] = node
            self.number_of_descendants_at_tree_depth[tree_depth] += 1
        else:
            self.descendants_at_tree_depth[tree_depth] = {node_tag: node}
            self.number_of_descendants_at_tree_depth[tree_depth] = 1
        self.number_of_descendants += 1

    def __len__(self) -> int:
        """
        Returns the number of descendants at the current half move.

        :return: The number of descendants at the current half move.
        :rtype: int
        """
        return len(self.descendants_at_tree_depth)

    def print_info(self) -> None:
        """
        Prints information about the descendants.

        This method prints the number of descendants and their corresponding half moves.
        It also prints the ID and fast representation of each descendant.

        Returns:
            None
        """
        print("---here are the ", self.get_count(), " descendants.")
        for tree_depth in self:
            print(
                "tree_depth: ",
                tree_depth,
                "| (",
                self.number_of_descendants_at_tree_depth[tree_depth],
                "descendants)",
            )  # ,                  end='| ')
            for descendant in self[tree_depth].values():
                print(descendant.id, descendant.tag, end=" ")
            print("")

    def print_stats(self) -> None:
        """
        Prints the statistics of the descendants.

        This method prints the number of descendants at each half move.

        Returns:
            None
        """
        print("---here are the ", self.get_count(), " descendants")
        for tree_depth in self:
            print(
                "tree_depth: ",
                tree_depth,
                "| (",
                self.number_of_descendants_at_tree_depth[tree_depth],
                "descendants)",
            )

    def test(self) -> None:
        """
        This method performs a series of assertions to validate the descendants data structure.
        It checks if the number of descendants at each half move matches the number of descendants stored.
        It also checks if the sum of the lengths of all descendants at each half move matches the total number of descendants.
        """
        assert set(self.descendants_at_tree_depth.keys()) == set(
            self.number_of_descendants_at_tree_depth
        )
        sum_ = 0
        for tree_depth in self:
            sum_ += len(self[tree_depth])
        assert self.number_of_descendants == sum_

        for tree_depth in self:
            assert self.number_of_descendants_at_tree_depth[tree_depth] == len(
                self[tree_depth]
            )


class RangedDescendants[TNode: ITreeNode[Any]](Descendants[TNode]):
    """
    Represents a collection of descendants with a range of half moves.

    Attributes:
        min_tree_depth (int | None): The minimum half move in the range.
        max_tree_depth (int | None): The maximum half move in the range.
    """

    min_tree_depth: int | None
    max_tree_depth: int | None

    def __init__(self) -> None:
        """
        Initializes a Descendants object.
        """
        super().__init__()
        self.min_tree_depth = None
        self.max_tree_depth = None

    def __str__(self) -> str:
        """
        Returns a string representation of the Descendants object.

        The string includes information about each half move and its descendants.

        Returns:
            str: A string representation of the Descendants object.
        """
        string: str = ""
        for tree_depth in self:
            string += f"tree_depth: {tree_depth} | ({self.number_of_descendants_at_tree_depth[tree_depth]} descendants)\n"
            for descendant in self[tree_depth].values():
                string += f"{descendant.id} "
            string += "\n"
        return string

    def is_new_generation(self, tree_depth: TreeDepth) -> bool:
        """
        Checks if the given half move is a new generation.

        Args:
            tree_depth (TreeDepth): The half move to check.

        Returns:
            bool: True if the half move is a new generation, False otherwise.
        """
        if self.min_tree_depth is not None:
            assert self.max_tree_depth is not None
            return tree_depth == self.max_tree_depth + 1
        else:
            return True

    def is_in_the_current_range(self, tree_depth: int) -> bool:
        """
        Checks if the given tree_depth is within the current range.

        Args:
            tree_depth (int): The tree_depth to check.

        Returns:
            bool: True if the tree_depth is within the range, False otherwise.
        """

        if self.min_tree_depth is not None and self.max_tree_depth is not None:
            return self.max_tree_depth >= tree_depth >= self.min_tree_depth
        else:
            return False

    def is_in_the_acceptable_range(self, tree_depth: int) -> bool:
        """
        Checks if the given tree_depth is within the acceptable range.

        Args:
            tree_depth (int): The tree_depth to check.

        Returns:
            bool: True if the tree_depth is within the acceptable range, False otherwise.
        """
        if self.min_tree_depth is not None and self.max_tree_depth is not None:
            return self.max_tree_depth + 1 >= tree_depth >= self.min_tree_depth
        else:
            return True

    def add_descendant(self, node: TNode) -> None:
        """
        Adds a descendant node to the tree.

        Args:
            node (TNode): The descendant node to be added.

        Returns:
            None
        """
        tree_depth: int = node.tree_depth
        node_tag: StateTag = node.tag

        assert self.is_in_the_acceptable_range(tree_depth)
        if self.is_in_the_current_range(tree_depth):
            if tree_depth in self.descendants_at_tree_depth:
                assert node_tag not in self.descendants_at_tree_depth[tree_depth]
            self.descendants_at_tree_depth[tree_depth][node_tag] = node
            self.number_of_descendants_at_tree_depth[tree_depth] += 1
        else:  # tree_depth == len(self.descendants_at_tree_depth)
            assert self.is_new_generation(tree_depth)
            self.descendants_at_tree_depth[tree_depth] = {node_tag: node}
            self.number_of_descendants_at_tree_depth[tree_depth] = 1
            if self.max_tree_depth is not None:
                self.max_tree_depth += 1
            else:
                self.min_tree_depth = tree_depth
                self.max_tree_depth = tree_depth
        self.number_of_descendants += 1

    def remove_descendant(self, node: TNode) -> None:
        """
        Removes a descendant node from the tree.

        Args:
            node (TNode): The node to be removed.

        Returns:
            None
        """
        tree_depth: int = node.tree_depth
        fen = node.tag

        self.number_of_descendants -= 1
        self[tree_depth].pop(fen)
        self.number_of_descendants_at_tree_depth[tree_depth] -= 1
        if self.number_of_descendants_at_tree_depth[tree_depth] == 0:
            self.number_of_descendants_at_tree_depth.pop(tree_depth)
            self.descendants_at_tree_depth.pop(tree_depth)
            if tree_depth == self.max_tree_depth:
                assert self.max_tree_depth is not None
                self.max_tree_depth -= 1
            if tree_depth == self.min_tree_depth:
                assert self.min_tree_depth is not None
                self.min_tree_depth += 1
            assert self.max_tree_depth is not None
            assert self.min_tree_depth is not None
            if self.max_tree_depth < self.min_tree_depth:
                self.max_tree_depth = None
                self.min_tree_depth = None
                assert self.number_of_descendants == 0

    def range(self) -> range:
        """
        Returns a range object representing the half moves range.

        The range starts from the minimum half move and ends at the maximum half move.

        Returns:
            range: A range object representing the half moves range.
        """

        assert self.max_tree_depth is not None
        assert self.min_tree_depth is not None
        return range(self.min_tree_depth, self.max_tree_depth + 1)

    #  def update(
    #          self,
    #          new_descendants: typing.Self
    #  ) -> RangedDescendants:
    #      really_new_descendants : RangedDescendants()#

    #        for tree_depth in new_descendants.range():
    #            if tree_depth in self:
    #               really_new_descendants_keys = set(new_descendants[tree_depth].keys()).difference(
    #                  set(self[tree_depth].keys()))
    #          else:
    #              really_new_descendants_keys = set(new_descendants[tree_depth].keys())
    #          for key in really_new_descendants_keys:
    #              really_new_descendants.add_descendant(new_descendants[tree_depth][key])
    #             self.add_descendant(new_descendants[tree_depth][key])

    #   # really_new_descendants.print_info()

    #    return really_new_descendants

    def merge(self, descendant_1: typing.Self, descendant_2: typing.Self) -> None:
        """
        Merges the descendants of two nodes into the current node.

        Args:
            descendant_1 (typing.Self): The first descendant node.
            descendant_2 (typing.Self): The second descendant node.

        Returns:
            None
        """
        tree_depths_range = set(descendant_1.keys()) | set(descendant_2.keys())
        assert len(tree_depths_range) > 0
        self.min_tree_depth = min(tree_depths_range)
        self.max_tree_depth = max(tree_depths_range)
        for tree_depth in tree_depths_range:
            if descendant_1.is_in_the_current_range(tree_depth):
                if descendant_2.is_in_the_current_range(tree_depth):
                    #  print('dd',type(self.descendants_at_tree_depth),type())
                    # in python 3.9 we can use a |
                    self.descendants_at_tree_depth[tree_depth] = {
                        **descendant_1[tree_depth],
                        **descendant_2[tree_depth],
                    }
                    self.number_of_descendants_at_tree_depth[tree_depth] = len(
                        self[tree_depth]
                    )
                    assert self.number_of_descendants_at_tree_depth[tree_depth] == len(
                        {**descendant_1[tree_depth], **descendant_2[tree_depth]}
                    )
                else:
                    self.descendants_at_tree_depth[tree_depth] = descendant_1[
                        tree_depth
                    ]
                    self.number_of_descendants_at_tree_depth[tree_depth] = (
                        descendant_1.number_of_descendants_at_tree_depth[tree_depth]
                    )
            else:
                self.descendants_at_tree_depth[tree_depth] = descendant_2[tree_depth]
                self.number_of_descendants_at_tree_depth[tree_depth] = (
                    descendant_2.number_of_descendants_at_tree_depth[tree_depth]
                )
            self.number_of_descendants += self.number_of_descendants_at_tree_depth[
                tree_depth
            ]

    def test(self) -> None:
        """
        Perform a test on the descendants object.

        This method checks the validity of the descendants object by asserting various conditions.
        If the `min_tree_depth` attribute is None, it asserts that `max_tree_depth` is also None and
        `number_of_descendants` is 0.
        Otherwise, it asserts that `max_tree_depth` and `min_tree_depth` are not None, and checks if all half moves
        between `min_tree_depth` and `max_tree_depth` are present in `descendants_at_tree_depth` dictionary.
        Finally, it iterates over all half moves in the descendants object and asserts that each half move
        is within the current range.

        Returns:
            None
        """
        super().test()
        if self.min_tree_depth is None:
            assert self.max_tree_depth is None
            assert self.number_of_descendants == 0
        else:
            assert self.max_tree_depth is not None
            assert self.min_tree_depth is not None
            for i in range(self.min_tree_depth, self.max_tree_depth + 1):
                assert i in self.descendants_at_tree_depth.keys()
        for tree_depth in self:
            assert self.is_in_the_current_range(tree_depth)

    def print_info(self) -> None:
        """
        Prints information about the descendants.

        This method calls the `print_info` method of the parent class and then prints the count of descendants,
        the minimum half move, and the maximum half move.

        Returns:
            None
        """
        super().print_info()
        print(
            "---here are the ",
            self.get_count(),
            " descendants. min:",
            self.min_tree_depth,
            ". max:",
            self.max_tree_depth,
        )


class SortedDescendants[TNode: ITreeNode[Any]](Descendants[TNode]):
    # todo is there a difference between sorted descendant nd sorted value descendant? below?

    """
    Represents a class that stores sorted descendants of a tree node at different half moves.
    Inherits from the Descendants class.
    """

    sorted_descendants_at_tree_depth: dict[int, dict[TNode, float]]

    def __init__(self) -> None:
        super().__init__()
        self.sorted_descendants_at_tree_depth = {}

    def update_value(self, node: TNode, value: float) -> None:
        """
        Updates the value of a descendant node.

        Args:
            node (TNode): The descendant node.
            value (float): The new value for the descendant node.
        """
        self.sorted_descendants_at_tree_depth[node.tree_depth][node] = value

    def add_descendant_with_val(self, node: TNode, value: float) -> None:
        """
        Adds a descendant node with its corresponding value.

        Args:
            node (TNode): The descendant node to add.
            value (float): The value of the descendant node.
        """
        super().add_descendant(node)
        tree_depth = node.tree_depth

        if tree_depth in self.sorted_descendants_at_tree_depth:
            assert node not in self.sorted_descendants_at_tree_depth[tree_depth]
            self.sorted_descendants_at_tree_depth[tree_depth][node] = value
        else:  # tree_depth == len(self.descendants_at_tree_depth)
            self.sorted_descendants_at_tree_depth[tree_depth] = {node: value}

        assert self.contains_node(node)

    def test(self) -> None:
        """
        Performs a test to ensure the integrity of the data structure.
        """
        super().test()
        assert len(self.sorted_descendants_at_tree_depth) == len(
            self.descendants_at_tree_depth
        )
        assert (
            self.sorted_descendants_at_tree_depth.keys()
            == self.descendants_at_tree_depth.keys()
        )
        for tree_depth in self.sorted_descendants_at_tree_depth:
            assert len(self.sorted_descendants_at_tree_depth[tree_depth]) == len(
                self.descendants_at_tree_depth[tree_depth]
            )

    def print_info(self) -> None:
        """
        Prints information about the sorted descendants.
        """
        super().print_info()
        print("sorted")
        for tree_depth in self:
            print(
                "tree_depth: ",
                tree_depth,
                "| (",
                self.number_of_descendants_at_tree_depth[tree_depth],
                "descendants)",
            )  # ,                  end='| ')
            for descendant, value in self.sorted_descendants_at_tree_depth[
                tree_depth
            ].items():
                print(descendant.id, descendant.tag, "(" + str(value) + ")", end=" ")
            print("")

    def remove_descendant(self, node: TNode) -> None:
        """
        Removes a descendant node from the data structure.

        Args:
            node (TNode): The descendant node to remove.
        """
        super().remove_descendant(node)
        tree_depth = node.tree_depth
        self.sorted_descendants_at_tree_depth[tree_depth].pop(node)
        if tree_depth not in self.number_of_descendants_at_tree_depth:
            self.sorted_descendants_at_tree_depth.pop(tree_depth)
        assert not self.contains_node(node)

    def contains_node(self, node: TNode) -> bool:
        """
        Checks if a descendant node is present in the data structure.

        Args:
            node (TNode): The descendant node to check.

        Returns:
            bool: True if the descendant node is present, False otherwise.
        """
        reply_base = super().contains_node(node)
        if (
            node.tree_depth in self.descendants_at_tree_depth
            and node in self.sorted_descendants_at_tree_depth[node.tree_depth]
        ):
            rep = True
        else:
            rep = False
        assert reply_base == rep
        return rep


class SortedValueDescendants[TNode: ITreeNode[Any]](Descendants[TNode]):
    """
    Represents a class for managing sorted descendants with associated values.
    Inherits from the `Descendants` class.
    """

    sorted_descendants_at_tree_depth: dict[TreeDepth, ValueSortedDict]

    def __init__(self) -> None:
        """
        Initializes a Sorted Value Descendants object.
        """
        super().__init__()
        self.sorted_descendants_at_tree_depth = {}

    def update_value(self, node: TNode, value: float) -> None:
        """
        Updates the value associated with a given node.

        Args:
            node (TNode): The node to update the value for.
            value (float): The new value to associate with the node.

        Returns:
            None
        """
        self.sorted_descendants_at_tree_depth[node.tree_depth][node] = value

    def add_descendant_val(self, node: TNode, value: float) -> None:
        """
        Adds a descendant node with an associated value.

        Args:
            node (TNode): The descendant node to add.
            value (float): The value associated with the descendant node.

        Returns:
            None
        """
        super().add_descendant(node)
        tree_depth = node.tree_depth

        if tree_depth in self.sorted_descendants_at_tree_depth:
            assert node not in self.sorted_descendants_at_tree_depth[tree_depth]
            self.sorted_descendants_at_tree_depth[tree_depth][node] = value
        else:
            self.sorted_descendants_at_tree_depth[tree_depth] = ValueSortedDict(
                {node: value}
            )

    def test(self) -> None:
        """
        Performs a test to ensure the integrity of the sorted descendants.

        Returns:
            None
        """
        super().test()
        assert len(self.sorted_descendants_at_tree_depth) == len(
            self.descendants_at_tree_depth
        )
        assert (
            self.sorted_descendants_at_tree_depth.keys()
            == self.descendants_at_tree_depth.keys()
        )
        for tree_depth in self.sorted_descendants_at_tree_depth:
            assert len(self.sorted_descendants_at_tree_depth[tree_depth]) == len(
                self.descendants_at_tree_depth[tree_depth]
            )

    def print_info(self) -> None:
        """
        Prints information about the sorted descendants.

        Returns:
            None
        """
        super().print_info()
        print("sorted")
        for tree_depth in self:
            print(
                "tree_depth: ",
                tree_depth,
                "| (",
                self.number_of_descendants_at_tree_depth[tree_depth],
                "descendants)",
            )
            for descendant, value in self.sorted_descendants_at_tree_depth[
                tree_depth
            ].items():
                print(str(descendant.id) + "(" + str(value) + ")", end=" ")
            print("")

    def remove_descendant(self, node: TNode) -> None:
        """
        Removes a descendant node.

        Args:
            node (TNode): The descendant node to remove.

        Returns:
            None
        """
        super().remove_descendant(node)
        tree_depth = node.tree_depth

        self.sorted_descendants_at_tree_depth[tree_depth].pop(node)
        if tree_depth not in self.number_of_descendants_at_tree_depth:
            self.sorted_descendants_at_tree_depth.pop(tree_depth)
