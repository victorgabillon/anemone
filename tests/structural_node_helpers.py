"""Helpers for lightweight structural node fakes in tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def make_structural_tree_node(
    *,
    node_id: int = 0,
    state: Any = None,
    branches_children: dict[Any, Any | None] | None = None,
    non_opened_branches: set[Any] | None = None,
    all_branches_generated: bool = True,
    **extra: Any,
) -> Any:
    """Return a mutable fake exposing the TreeNode structural API."""
    child_links = branches_children if branches_children is not None else {}
    unopened = non_opened_branches if non_opened_branches is not None else set()
    node = SimpleNamespace(
        id=node_id,
        state=state if state is not None else SimpleNamespace(),
        branches_children=child_links,
        non_opened_branches=unopened,
        all_branches_generated=all_branches_generated,
        **extra,
    )

    def iter_child_links() -> Iterator[tuple[Any, Any | None]]:
        return iter(node.branches_children.items())

    def iter_child_nodes() -> Iterator[Any]:
        return (child for child in node.branches_children.values() if child is not None)

    def child_for_branch(branch: Any) -> Any | None:
        return node.branches_children.get(branch)

    def set_child_for_branch(branch: Any, child: Any | None) -> None:
        node.branches_children[branch] = child

    def discard_unopened_branch(branch: Any) -> None:
        node.non_opened_branches.discard(branch)

    def set_unopened_branches(branches: Iterable[Any]) -> None:
        node.non_opened_branches = set(branches)

    node.iter_child_links = iter_child_links
    node.iter_child_nodes = iter_child_nodes
    node.child_for_branch = child_for_branch
    node.has_child_links = lambda: bool(node.branches_children)
    node.child_link_count = lambda: len(node.branches_children)
    node.has_child_link_for_branch = lambda branch: branch in node.branches_children
    node.has_concrete_child_for_branch = lambda branch: (
        node.branches_children.get(branch) is not None
    )
    node.has_child_for_branch = node.has_concrete_child_for_branch
    node.set_child_for_branch = set_child_for_branch
    node.discard_unopened_branch = discard_unopened_branch
    node.set_unopened_branches = set_unopened_branches
    node.iter_unopened_branches = lambda: iter(node.non_opened_branches)
    node.has_unopened_branches = lambda: bool(node.non_opened_branches)
    node.unopened_branch_count = lambda: len(node.non_opened_branches)
    return node
