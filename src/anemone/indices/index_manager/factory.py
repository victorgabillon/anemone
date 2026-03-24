"""Create exploration-index update strategies for configured search modes."""

from anemone.indices.index_manager.node_exploration_manager import (
    NodeExplorationIndexManager,
    NullNodeExplorationIndexManager,
    UpdateIndexGlobalMinChange,
    UpdateIndexLocalMinChange,
    UpdateIndexZipfFactoredProba,
)
from anemone.indices.node_indices.index_types import (
    IndexComputationType,
)


class UnknownIndexComputationError(ValueError):
    """Raised when an exploration-index computation mode is not recognized."""

    def __init__(self, index_computation: IndexComputationType) -> None:
        """Initialize the error with the unsupported computation type."""
        super().__init__(
            f"Unsupported exploration-index computation mode: {index_computation!s}."
        )


def create_exploration_index_manager(
    index_computation: IndexComputationType | None = None,
) -> NodeExplorationIndexManager:
    """Create the exploration-index update strategy for one computation mode.

    Args:
        index_computation: Configured exploration-index computation mode. ``None``
            selects the null/no-op strategy.

    Returns:
        The coordinator/strategy object responsible for recomputing exploration
        payloads.

    Raises:
        UnknownIndexComputationError: If the computation mode is unsupported.

    """
    node_exploration_manager: NodeExplorationIndexManager
    if index_computation is None:
        node_exploration_manager = NullNodeExplorationIndexManager()
    else:
        match index_computation:
            case IndexComputationType.MIN_GLOBAL_CHANGE:
                node_exploration_manager = UpdateIndexGlobalMinChange()
            case IndexComputationType.RECUR_ZIPF:
                node_exploration_manager = UpdateIndexZipfFactoredProba()
            case IndexComputationType.MIN_LOCAL_CHANGE:
                node_exploration_manager = UpdateIndexLocalMinChange()
            case _:
                raise UnknownIndexComputationError(index_computation)

    return node_exploration_manager
