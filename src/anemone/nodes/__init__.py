"""
This module contains the implementation of tree nodes for branch selection.

The tree nodes are used in the branch selector to represent different branches and their values.

Classes:
- TreeNode: Represents a tree node for branch selection.
- ITreeNode: Interface for tree nodes.

"""

from .itree_node import ITreeNode
from .tree_node import TreeNode

__all__ = ["ITreeNode", "TreeNode"]
