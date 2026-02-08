"""Provide a factory for node exploration index managers."""

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
    """Raised when an index computation type is not recognized."""

    def __init__(self, index_computation: IndexComputationType) -> None:
        """Initialize the error with the unsupported computation type."""
        super().__init__(
            f"player creator: can not find {index_computation} in {__name__}"
        )


def create_exploration_index_manager(
    index_computation: IndexComputationType | None = None,
) -> NodeExplorationIndexManager:
    """Create a node exploration index manager for the given index computation type.

    Args:
        index_computation (IndexComputationType | None): The type of index computation to be used.
        Defaults to None.

    Returns:
        NodeExplorationIndexManager: The created node exploration index manager.

    Raises:
        ValueError: If the given index computation type is not found.

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
