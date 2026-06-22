"""Structural helpers for partially opened tree nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from valanga import BranchKey

    from anemone.dynamics import SearchDynamics
    from anemone.nodes.itree_node import ITreeNode


def _iter_child_links(
    node: ITreeNode[Any],
) -> Iterable[tuple[BranchKey, ITreeNode[Any] | None]]:
    """Iterate child links, preferring the structural API when available."""
    iter_child_links = getattr(node, "iter_child_links", None)
    if callable(iter_child_links):
        return cast(
            "Iterable[tuple[BranchKey, ITreeNode[Any] | None]]", iter_child_links()
        )
    return cast(
        "Iterable[tuple[BranchKey, ITreeNode[Any] | None]]",
        node.branches_children.items(),
    )


def opened_branch_keys(node: ITreeNode[Any]) -> set[BranchKey]:
    """Return branch keys with concrete child links from ``node``."""
    return {
        branch_key for branch_key, child in _iter_child_links(node) if child is not None
    }


def legal_branch_keys(
    *,
    node: ITreeNode[Any],
    dynamics: SearchDynamics[Any, Any],
) -> Sequence[BranchKey]:
    """Return legal branches in dynamics-provided order."""
    return dynamics.legal_actions(node.state).get_all()


def openable_branch_keys(
    *,
    node: ITreeNode[Any],
    dynamics: SearchDynamics[Any, Any],
) -> tuple[BranchKey, ...]:
    """Return legal branches that do not yet have concrete child links."""
    opened = opened_branch_keys(node)
    return tuple(
        branch_key
        for branch_key in legal_branch_keys(node=node, dynamics=dynamics)
        if branch_key not in opened
    )


def has_openable_branches(
    *,
    node: ITreeNode[Any],
    dynamics: SearchDynamics[Any, Any],
) -> bool:
    """Return whether ``node`` still has legal unopened branches."""
    return bool(openable_branch_keys(node=node, dynamics=dynamics))


def is_fully_open_wrt_legal_actions(
    *,
    node: ITreeNode[Any],
    dynamics: SearchDynamics[Any, Any],
) -> bool:
    """Return whether every legal branch already has a concrete child link."""
    return not has_openable_branches(node=node, dynamics=dynamics)


def sync_opening_status(
    *,
    node: ITreeNode[Any],
    dynamics: SearchDynamics[Any, Any],
) -> None:
    """Synchronize opening bookkeeping from actual legal/opened branch state."""
    openable = openable_branch_keys(node=node, dynamics=dynamics)
    node.all_branches_generated = not openable
    set_unopened_branches = getattr(node, "set_unopened_branches", None)
    if callable(set_unopened_branches):
        set_unopened_branches(openable)
        return
    node.non_opened_branches.clear()
    node.non_opened_branches.update(openable)
