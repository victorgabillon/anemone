"""
Module for DictOfNumberedDictWithPointerOnMax class.
"""

from typing import Protocol


class HasTreeDepth(Protocol):
    """Protocol for objects that have a tree_depth attribute."""

    @property
    def tree_depth(self) -> int:
        """
        Get the tree depth of the node.

        Returns:
            The tree depth of the node.
        """
        ...


class DictOfNumberedDictWithPointerOnMax[T_Key: HasTreeDepth, T_Value]:
    """
    A dictionary-like data structure that stores numbered dictionaries and keeps track of the maximum depth.

    Attributes:
        tree_depths (dict[int, dict[T_Key, T_Value]]): A dictionary that stores numbered dictionaries.
        max_tree_depth (int | None): The maximum depth value.

    Methods:
        __setitem__(self, node: T_Key, value: T_Value) -> None: Adds an item to the data structure.
        __getitem__(self, node: T_Key) -> T_Value: Retrieves an item from the data structure.
        __bool__(self) -> bool: Checks if the data structure is non-empty.
        __contains__(self, node: T_Key) -> bool: Checks if an item is present in the data structure.
        popitem(self) -> tuple[T_Key, T_Value]: Removes and returns the item with the maximum depth value.
    """

    def __init__(self) -> None:
        """Initialize the depth-indexed mapping."""
        self.tree_depths: dict[int, dict[T_Key, T_Value]] = {}
        self.max_tree_depth: int | None = None

    def __setitem__(self, node: T_Key, value: T_Value) -> None:
        """
        Adds an item to the data structure.

        Args:
            node (T_Key): The key of the item.
            value (T_Value): The value of the item.

        Returns:
            None
        """
        tree_depth = node.tree_depth
        if self.max_tree_depth is None:
            self.max_tree_depth = tree_depth
        else:
            self.max_tree_depth = max(tree_depth, self.max_tree_depth)
        if tree_depth in self.tree_depths:
            self.tree_depths[tree_depth][node] = value
        else:
            self.tree_depths[tree_depth] = {node: value}

        assert self.max_tree_depth == max(self.tree_depths)

    def __getitem__(self, node: T_Key) -> T_Value:
        """
        Retrieves an item from the data structure.

        Args:
            node (T_Key): The key of the item.

        Returns:
            T_Value: The value of the item.

        Raises:
            KeyError: If the item is not found in the data structure.
        """
        return self.tree_depths[node.tree_depth][node]

    def __bool__(self) -> bool:
        """
        Checks if the data structure is non-empty.

        Returns:
            bool: True if the data structure is non-empty, False otherwise.
        """
        return bool(self.tree_depths)

    def __contains__(self, node: T_Key) -> bool:
        """
        Checks if an item is present in the data structure.

        Args:
            node (T_Key): The key of the item.

        Returns:
            bool: True if the item is present, False otherwise.
        """
        if node.tree_depth not in self.tree_depths:
            return False
        else:
            return node in self.tree_depths[node.tree_depth]

    def popitem(self) -> tuple[T_Key, T_Value]:
        """
        Removes and returns the item with the maximum depth value.

        Returns:
            tuple[T_Key, T_Value]: The key-value pair of the removed item.

        Raises:
            AssertionError: If the data structure is empty.
        """
        assert self.max_tree_depth is not None
        popped: tuple[T_Key, T_Value] = self.tree_depths[self.max_tree_depth].popitem()
        if not self.tree_depths[self.max_tree_depth]:
            del self.tree_depths[self.max_tree_depth]
            if self.tree_depths:
                self.max_tree_depth = max(self.tree_depths.keys())
            else:
                self.max_tree_depth = None

        return popped

    # def sort_dic(self):
    #    self.dic = dict(sorted(self.dic.items(), key=lambda item: item[0]))
    # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
