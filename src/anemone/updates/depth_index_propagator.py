"""Incremental descendant-depth propagation via dirty ancestors and depth waves.

This module restores the live ``MaxDepthDescendants`` feature without bringing
back the removed mixed value/index instruction pipeline. It owns only the
scheduling of dirty ancestors for descendant-depth metadata.

Invariants encoded by ``DepthIndexPropagator``:

1. A node's descendant-depth metadata is recomputed from a complete snapshot of
   its current children.
2. Child changes do not patch parents directly; they only mark ancestors dirty.
3. Dirty nodes are processed in descending depth order so deeper structural
   changes stabilize before shallower ancestors are recomputed.
4. A node is recomputed at most once per depth wave, even if several changed
   children mark it dirty in that wave.

This propagator is intentionally separate from both value propagation and
exploration-index refresh. It maintains only ``MaxDepthDescendants`` metadata.
"""

from collections.abc import Callable, Iterable
from typing import Any

from anemone.indices.node_indices.index_data import MaxDepthDescendants
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

RecomputeDepthIndex = Callable[[AlgorithmNode[Any]], bool]


class DepthIndexPropagator:
    """Propagate ``MaxDepthDescendants`` updates through dirty ancestor waves."""

    def __init__(
        self,
        recompute_node_depth_index: RecomputeDepthIndex | None = None,
    ) -> None:
        """Create a propagator with an optional injected recomputation hook."""
        self._recompute_node_depth_index_impl = (
            recompute_node_depth_index
            or self._recompute_node_depth_index_from_full_child_snapshot
        )

    def propagate_from_changed_nodes(
        self,
        changed_nodes: Iterable[AlgorithmNode[Any]],
    ) -> set[AlgorithmNode[Any]]:
        """Incrementally recompute dirty ancestors until descendant depth settles.

        Args:
            changed_nodes: Nodes whose structural descendant-depth view just
                changed locally, typically newly created children or children
                linked to a new parent.

        Returns:
            The set of ancestor nodes touched/recomputed during propagation.

        """
        dirty_by_depth: dict[int, set[AlgorithmNode[Any]]] = {}
        touched_nodes: set[AlgorithmNode[Any]] = set()

        for changed_node in changed_nodes:
            self._mark_parents_dirty(
                changed_node=changed_node,
                dirty_by_depth=dirty_by_depth,
            )

        # Process deeper dirty nodes first so shallower parents observe already
        # stabilized descendant-depth metadata from the current wave.
        while dirty_by_depth:
            deepest_depth = max(dirty_by_depth)
            dirty_nodes_at_depth = dirty_by_depth.pop(deepest_depth)
            changed_this_wave: set[AlgorithmNode[Any]] = set()

            for node in dirty_nodes_at_depth:
                touched_nodes.add(node)
                if self._recompute_node_depth_index_impl(node):
                    changed_this_wave.add(node)

            for node in changed_this_wave:
                self._mark_parents_dirty(
                    changed_node=node,
                    dirty_by_depth=dirty_by_depth,
                )

        return touched_nodes

    def _mark_parents_dirty(
        self,
        *,
        changed_node: AlgorithmNode[Any],
        dirty_by_depth: dict[int, set[AlgorithmNode[Any]]],
    ) -> None:
        """Mark each parent dirty in the bucket matching the parent's depth."""
        parent_node: AlgorithmNode[Any]
        for parent_node in changed_node.parent_nodes:
            dirty_by_depth.setdefault(parent_node.tree_depth, set()).add(parent_node)

    def _recompute_node_depth_index_from_full_child_snapshot(
        self,
        node: AlgorithmNode[Any],
    ) -> bool:
        """Recompute one node's descendant-depth metadata from all current children.

        ``MaxDepthDescendants.update_from_child(...)`` is still the semantic core
        for combining child depths. This adapter resets the node to the neutral
        value and replays every currently linked child so the result always comes
        from a complete child snapshot rather than a partial child delta.
        """
        exploration_index_data = node.exploration_index_data
        if not isinstance(exploration_index_data, MaxDepthDescendants):
            return False

        previous_max_depth = exploration_index_data.max_depth_descendants
        exploration_index_data.max_depth_descendants = 0

        child_node: AlgorithmNode[Any] | None
        for child_node in node.branches_children.values():
            if child_node is None:
                continue
            exploration_index_data.update_from_child(
                self._child_max_depth_descendants(child_node)
            )

        return exploration_index_data.max_depth_descendants != previous_max_depth

    def _child_max_depth_descendants(self, child_node: AlgorithmNode[Any]) -> int:
        """Return the child's currently known descendant depth.

        Depth-indexed trees are expected to attach ``MaxDepthDescendants`` data
        to all participating nodes. When that metadata is absent, we conservatively
        treat the child as a leaf so the propagator degrades harmlessly instead
        of reviving the removed mixed updater path.
        """
        child_index_data = child_node.exploration_index_data
        if isinstance(child_index_data, MaxDepthDescendants):
            return child_index_data.max_depth_descendants
        return 0
