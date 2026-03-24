"""Public coordination surface for exploration-index update strategies.

Use ``create_exploration_index_manager(...)`` to choose the strategy/coordinator
for one search configuration, then ``update_all_indices(...)`` to recompute the
current exploration payloads across a tree.
"""

from .factory import create_exploration_index_manager
from .node_exploration_manager import (
    NodeExplorationIndexManager,
    NullNodeExplorationIndexManager,
    update_all_indices,
)

__all__ = [
    "NodeExplorationIndexManager",
    "NullNodeExplorationIndexManager",
    "create_exploration_index_manager",
    "update_all_indices",
]
