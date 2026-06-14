"""Provide functionality for managing algorithm node trees.

The module includes classes for creating and managing algorithm node tree managers,
expanding tree structures, and managing tree expansions.

Classes:
- AlgorithmNodeTreeManager: A class for managing algorithm node trees.
- TreeManager: A class for managing tree structures.
- TreeExpansion: A class for representing a single tree expansion.
- TreeExpansions: A class for managing multiple tree expansions.

Functions:
- create_algorithm_node_tree_manager: A function for creating an algorithm node tree manager.

"""

from .algorithm_node_tree_manager import AlgorithmNodeTreeManager
from .branch_opening_service import BranchOpeningService, BranchOpeningTreeManager
from .factory import create_algorithm_node_tree_manager
from .opening_expansion_budget import OpeningExpansionBudget
from .opening_expansion_config import (
    OpeningExpansionConfig,
    OpeningExpansionKind,
    RolloutActionSelectorKind,
    RolloutExpansionConfig,
)
from .opening_expansion_executor import (
    OnePlyOpeningExpansionExecutor,
    OpeningExpansionExecutor,
)
from .opening_expansion_factory import (
    create_opening_expansion_executor,
    create_rollout_action_selector,
)
from .tree_expander import TreeExpansion, TreeExpansions
from .tree_manager import DuplicateBranchOpenError, TreeManager

__all__ = [
    "AlgorithmNodeTreeManager",
    "BranchOpeningService",
    "BranchOpeningTreeManager",
    "DuplicateBranchOpenError",
    "OnePlyOpeningExpansionExecutor",
    "OpeningExpansionBudget",
    "OpeningExpansionConfig",
    "OpeningExpansionExecutor",
    "OpeningExpansionKind",
    "RolloutActionSelectorKind",
    "RolloutExpansionConfig",
    "TreeExpansion",
    "TreeExpansions",
    "TreeManager",
    "create_algorithm_node_tree_manager",
    "create_opening_expansion_executor",
    "create_rollout_action_selector",
]
