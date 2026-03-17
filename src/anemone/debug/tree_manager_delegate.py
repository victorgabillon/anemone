"""Small helpers for optional tree-manager delegation wrappers."""

from typing import Any


def refresh_exploration_indices(base: Any, *, tree: Any) -> None:
    """Call the modern refresh hook when available, else use the legacy alias."""
    refresh = getattr(base, "refresh_exploration_indices", None)
    if refresh is not None:
        refresh(tree=tree)
        return
    base.update_indices(tree=tree)


def propagate_depth_index(base: Any, *, tree_expansions: Any) -> Any:
    """Delegate descendant-depth propagation only when the base exposes it."""
    propagate = getattr(base, "propagate_depth_index", None)
    if propagate is None:
        return None
    return propagate(tree_expansions=tree_expansions)
