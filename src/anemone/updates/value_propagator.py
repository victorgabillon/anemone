"""Incremental value propagation built around dirty ancestors and depth waves.

This module intentionally introduces scheduling only. It does not change backup
semantics, selector behavior, or exploration-index refresh. Dirty children never
patch parents directly; they only mark parents dirty. Each dirty parent is then
recomputed from a complete snapshot of its current children and local value
state by delegating to the existing node backup logic.

Invariants encoded by ``ValuePropagator``:

1. A node's value is recomputed only from a complete snapshot of its current
   children and local value state.
2. Child changes do not directly patch parent values. They only mark ancestors
   dirty.
3. Dirty nodes are processed in descending depth order so deeper value changes
   stabilize before shallower ancestors are recomputed. This matters because
   nodes may have multiple parents and multiple siblings can change in the same
   propagation wave.
4. A node is recomputed at most once per depth wave, even if several changed
   children mark it dirty in that wave.

The returned set of affected nodes is intentionally value-only. Callers may use
it later when deciding whether and how to refresh indices, but index updates are
deliberately out of scope here.
"""

from collections.abc import Callable, Iterable
from typing import Any

from valanga import BranchKey

from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

RecomputeNodeValue = Callable[[AlgorithmNode[Any]], bool]


class ValuePropagator:
    """Propagate dirty ancestor value recomputations in descending depth waves."""

    def __init__(
        self,
        recompute_node_value: RecomputeNodeValue | None = None,
    ) -> None:
        """Create a propagator with an optional injected node recomputation hook.

        The propagator owns only scheduling. The injected callable must
        recompute one node's value using the existing backup semantics and return
        whether the change is parent-relevant.
        """
        self._recompute_node_value_impl = (
            recompute_node_value
            or self._recompute_node_value_from_full_child_snapshot
        )

    def propagate_from_changed_nodes(
        self,
        changed_nodes: Iterable[AlgorithmNode[Any]],
    ) -> set[AlgorithmNode[Any]]:
        """Incrementally recompute dirty ancestors until the value wave settles.

        Args:
            changed_nodes: Nodes whose local or backed-up value just changed.

        Returns:
            The set of ancestor nodes touched/recomputed during propagation.
            This is intentionally broader than "actually changed nodes".
        """
        dirty_by_depth: dict[int, set[AlgorithmNode[Any]]] = {}
        touched_nodes: set[AlgorithmNode[Any]] = set()

        for changed_node in changed_nodes:
            self._mark_parents_dirty(
                changed_node=changed_node,
                dirty_by_depth=dirty_by_depth,
            )

        # Process deepest dirty nodes first so every shallower recomputation sees
        # already-stabilized child values for the current wave.
        while dirty_by_depth:
            deepest_depth = max(dirty_by_depth)
            dirty_nodes_at_depth = dirty_by_depth.pop(deepest_depth)
            changed_this_wave: set[AlgorithmNode[Any]] = set()

            for node in dirty_nodes_at_depth:
                touched_nodes.add(node)
                if self._recompute_node_value_impl(node):
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

    def _recompute_node_value_from_full_child_snapshot(
        self,
        node: AlgorithmNode[Any],
    ) -> bool:
        """Delegate one node recomputation to the existing backup semantics.

        The existing backup surface expects sets of updated child branches. For
        full-snapshot recomputation we intentionally pass *all* currently-open
        child branches, never a partial child-delta payload. This keeps the
        propagator focused on scheduling while preserving the repository's
        current value semantics.
        """
        current_child_branches = self._current_child_branches(node)
        if not current_child_branches:
            return False

        backup_result = node.tree_evaluation.backup_from_children(
            branches_with_updated_value=current_child_branches,
            branches_with_updated_best_branch_seq=current_child_branches,
        )
        # ``over_changed`` currently flows through canonical Value metadata and
        # is therefore already reflected by ``value_changed`` for parent-facing
        # value propagation.
        # We also treat PV changes as parent-relevant for now. That may
        # over-schedule slightly, but it avoids missing ancestor recomputations
        # while the propagator is still an additive internal component.
        return backup_result.value_changed or backup_result.pv_changed

    def _current_child_branches(
        self,
        node: AlgorithmNode[Any],
    ) -> set[BranchKey]:
        """Return the full snapshot of currently-open child branches."""
        return {
            branch_key
            for branch_key, child in node.branches_children.items()
            if child is not None
        }
