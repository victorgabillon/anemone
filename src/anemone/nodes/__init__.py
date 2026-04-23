"""Public node-layer vocabulary for structural and runtime-facing tree nodes.

Preferred meanings:

* ``ITreeNode``: structural/navigation protocol
* ``TreeNode``: concrete structural implementation
* ``AlgorithmNode``: runtime/search wrapper in
  ``anemone.nodes.algorithm_node`` that adds evaluation and exploration state
  on top of ``TreeNode``

This package re-exports the structural surfaces only; import
``AlgorithmNode`` from ``anemone.nodes.algorithm_node`` when you need the
runtime wrapper explicitly.
"""

from .itree_node import ITreeNode
from .state_handles import (
    CheckpointBackedStateHandle,
    CheckpointStateResolver,
    MaterializedStateHandle,
    StateHandle,
)
from .tree_node import TreeNode

__all__ = [
    "CheckpointBackedStateHandle",
    "CheckpointStateResolver",
    "ITreeNode",
    "MaterializedStateHandle",
    "StateHandle",
    "TreeNode",
]
