"""Shared dirty-ancestor wave scheduling for incremental upward propagators."""

from collections.abc import Callable, Iterable
from typing import Any

from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

type RecomputeNode[_NodeT: AlgorithmNode[Any]] = Callable[[_NodeT], bool]


def propagate_dirty_ancestors[NodeT: AlgorithmNode[Any]](
    changed_nodes: Iterable[NodeT],
    *,
    recompute_node: RecomputeNode[NodeT],
) -> set[NodeT]:
    """Recompute dirty ancestors in descending depth waves.

    Child-local changes seed the first dirty-parent wave. Each recomputed node
    that changes then seeds the next parent wave, allowing upward propagation
    to stabilize from deeper nodes toward the root.
    """
    dirty_by_depth: dict[int, set[NodeT]] = {}
    touched_nodes: set[NodeT] = set()

    for changed_node in changed_nodes:
        mark_parent_nodes_dirty(
            changed_node=changed_node,
            dirty_by_depth=dirty_by_depth,
        )

    while dirty_by_depth:
        deepest_depth = max(dirty_by_depth)
        dirty_nodes_at_depth = dirty_by_depth.pop(deepest_depth)
        changed_this_wave: set[NodeT] = set()

        for node in dirty_nodes_at_depth:
            touched_nodes.add(node)
            if recompute_node(node):
                changed_this_wave.add(node)

        for node in changed_this_wave:
            mark_parent_nodes_dirty(
                changed_node=node,
                dirty_by_depth=dirty_by_depth,
            )

    return touched_nodes


def mark_parent_nodes_dirty[NodeT: AlgorithmNode[Any]](
    *,
    changed_node: NodeT,
    dirty_by_depth: dict[int, set[NodeT]],
) -> None:
    """Bucket each parent according to its tree depth."""
    parent_node: NodeT
    for parent_node in changed_node.parent_nodes:
        dirty_by_depth.setdefault(parent_node.tree_depth, set()).add(parent_node)
