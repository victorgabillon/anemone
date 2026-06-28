"""Single-player depth-aware selector based on direct node values."""

# pylint: disable=duplicate-code

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone.node_evaluation.common.value_candidate import (
    ValueCandidate,
    ValueCandidateSource,
)
from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
    create_instructions_to_open_all_branches,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.objectives import SingleAgentMaxObjective

from .candidate_heap import (
    LinooCandidateHeap as _LinooCandidateHeap,
)
from .candidate_heap import (
    LinooHeapEntry as _LinooHeapEntry,
)
from .candidate_heap import (
    LinooHeapUpdateDiagnostics as _LinooHeapUpdateDiagnostics,
)
from .candidate_heap import (
    LinooRankedCandidate as _LinooRankedCandidate,
)
from .checkpoint_state import (
    candidate_payloads_from_cache,
    depth_stats_from_tree_and_node_states,
    depth_stats_match_payload,
    depth_stats_payload_from_cache,
    frontier_ids_from_node_states,
    node_states_payload_from_cache,
    payload_last_selected_node_id,
    restore_candidate_payloads,
    restore_node_states_from_payload,
)
from .depth_policy import (
    choose_depth,
    depth_selection_row_sort_key,
    inverse_depth_weight,
)
from .errors import (
    LinooDirectValueUnavailableError,
    LinooIncompatibleObjectiveError,
    LinooSelectionError,
)
from .errors import (
    incompatible_objective_error as _incompatible_objective_error,
)
from .errors import (
    invalid_linoo_checkpoint_payload_error as _invalid_linoo_checkpoint_payload_error,
)
from .errors import (
    invalid_linoo_depth_selection_policy_error as _invalid_linoo_depth_selection_policy_error,
)
from .errors import (
    missing_direct_value_attribute_error as _missing_direct_value_attribute_error,
)
from .errors import (
    missing_direct_value_error as _missing_direct_value_error,
)
from .errors import (
    missing_root_objective_error as _missing_root_objective_error,
)
from .errors import (
    no_frontier_nodes_error as _no_frontier_nodes_error,
)
from .report import LinooDepthSelectionRow, LinooSelectionReport
from .runtime_state import (
    LinooDepthStats as _LinooDepthStats,
)
from .runtime_state import (
    LinooNodeState as _LinooNodeState,
)
from .types import (
    LINOO_DEFAULT_NODE_STATUS as _LINOO_DEFAULT_NODE_STATUS,
)
from .types import (
    LinooArgs,
    LinooDepthSelectionPolicy,
)
from .types import (
    LinooNodeStatus as _LinooNodeStatus,
)

__all__ = [
    "Linoo",
    "LinooArgs",
    "LinooDepthSelectionPolicy",
    "LinooDepthSelectionRow",
    "LinooDirectValueUnavailableError",
    "LinooIncompatibleObjectiveError",
    "LinooSelectionError",
    "LinooSelectionReport",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager
    from random import Random

    from valanga.evaluations import Value

    from anemone import tree_manager as tree_man
    from anemone import trees
    from anemone.checkpoints.payloads import LinooSelectorCheckpointPayload


class Linoo[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Depth-aware single-player selector using frontier direct values.

    Linoo first selects a frontier depth using its configured depth policy, then
    samples an unopened node at that depth using Zipf rank weights over direct
    priorities.

    Cache invariant: direct values of existing frontier nodes are assumed stable
    between selections unless the selector cache is invalidated or the node is
    part of the latest expansion wave.
    """

    opening_instructor: OpeningInstructor
    random_generator: Random
    depth_selection_policy: LinooDepthSelectionPolicy
    latest_selection_report: LinooSelectionReport | None
    _depth_stats_by_depth: dict[int, _LinooDepthStats]
    _frontier_node_ids_by_depth: dict[int, set[int]]
    _node_state_by_id: dict[int, _LinooNodeState]
    _candidates_by_depth: dict[int, list[_LinooHeapEntry]]
    _candidate_versions_by_node_id: dict[int, int]
    _candidate_signature_by_node_id: dict[int, object]
    _candidate_heap_present_by_node_id: dict[int, bool]
    _candidate_heap: _LinooCandidateHeap
    _cache_tree_identity: int | None
    _cache_objective_identity: int | None
    _cache_initialized: bool
    _last_selected_node_id: int | None

    def __init__(
        self,
        opening_instructor: OpeningInstructor,
        random_generator: Random,
        *,
        depth_selection_policy: LinooDepthSelectionPolicy = "inverse_depth",
    ) -> None:
        """Store the opening instructor used to materialize branch openings."""
        if depth_selection_policy not in (
            "inverse_depth",
            "opened_count_depth_index",
        ):
            raise _invalid_linoo_depth_selection_policy_error(depth_selection_policy)
        self.opening_instructor = opening_instructor
        self.random_generator = random_generator
        self.depth_selection_policy = depth_selection_policy
        self.latest_selection_report = None
        self._depth_stats_by_depth = {}
        self._frontier_node_ids_by_depth = {}
        self._node_state_by_id = {}
        self._candidate_heap = _LinooCandidateHeap()
        self._candidates_by_depth = self._candidate_heap.candidates_by_depth
        self._candidate_versions_by_node_id = (
            self._candidate_heap.candidate_versions_by_node_id
        )
        self._candidate_signature_by_node_id = (
            self._candidate_heap.candidate_signature_by_node_id
        )
        self._candidate_heap_present_by_node_id = (
            self._candidate_heap.candidate_heap_present_by_node_id
        )
        self._cache_tree_identity = None
        self._cache_objective_identity = None
        self._cache_initialized = False
        self._last_selected_node_id = None

    @contextmanager
    def _diagnostic_phase(self, phase: str) -> Generator[None]:
        """Enter an optional external diagnostic phase context."""
        phase_context = cast(
            "Callable[[str], AbstractContextManager[None]] | None",
            getattr(self, "_diagnostic_phase_context", None),
        )
        if phase_context is not None:
            with phase_context(phase):
                yield
            return
        yield

    def invalidate(self) -> None:
        """Discard incremental selector state so the next selection rebuilds."""
        self._clear_runtime_state()

    def refresh_state_for_checkpoint(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
        latest_tree_expansions: tree_man.TreeExpansions[NodeT],
    ) -> None:
        """Refresh initialized runtime cache before checkpoint serialization."""
        if not self._cache_initialized:
            return
        self._ensure_runtime_state(
            tree=tree,
            objective=objective,
            latest_tree_expansions=latest_tree_expansions,
        )

    def build_checkpoint_payload(
        self,
        objective: SingleAgentMaxObjective[Any],
    ) -> LinooSelectorCheckpointPayload | None:
        """Return checkpoint payload for initialized Linoo runtime state."""
        from anemone.checkpoints.payloads import (  # pylint: disable=import-outside-toplevel
            LinooSelectorCheckpointPayload,
        )

        del objective
        if not self._cache_initialized:
            return None
        return LinooSelectorCheckpointPayload(
            depth_stats=depth_stats_payload_from_cache(self._depth_stats_by_depth),
            node_states=node_states_payload_from_cache(self._node_state_by_id),
            candidates_by_depth=candidate_payloads_from_cache(
                candidates_by_depth=self._candidates_by_depth,
                node_state_by_id=self._node_state_by_id,
                candidate_versions_by_node_id=self._candidate_versions_by_node_id,
            ),
            last_selected_node_id=self._last_selected_node_id,
        )

    def restore_from_checkpoint_payload(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
        payload: LinooSelectorCheckpointPayload,
    ) -> bool:
        """Restore Linoo runtime state from checkpoint payload.

        Return ``True`` when restored. Return ``False`` when optional selector
        state was stale or invalid and the next selection should rebuild.
        """
        if payload.type != "linoo" or payload.version != 1:
            return False
        nodes_by_id = self._nodes_by_id_from_tree(tree)
        try:
            node_states = restore_node_states_from_payload(
                tree=tree,
                nodes_by_id=nodes_by_id,
                payload=payload,
                classify_node=self._classify_node,
            )
            depth_stats = depth_stats_from_tree_and_node_states(
                tree=tree,
                nodes_by_id=nodes_by_id,
                node_states=node_states,
            )
            if not depth_stats_match_payload(depth_stats, payload.depth_stats):
                raise _invalid_linoo_checkpoint_payload_error()
            frontier_node_ids = frontier_ids_from_node_states(node_states)
            last_selected_node_id = payload_last_selected_node_id(payload)
            if (
                last_selected_node_id is not None
                and last_selected_node_id not in nodes_by_id
            ):
                raise _invalid_linoo_checkpoint_payload_error()
        except (KeyError, ValueError):
            self.invalidate()
            return False

        self._clear_runtime_state()
        self._depth_stats_by_depth = depth_stats
        self._frontier_node_ids_by_depth = frontier_node_ids
        self._node_state_by_id = node_states
        self._cache_tree_identity = id(tree.root_node)
        self._cache_objective_identity = id(objective)
        self._cache_initialized = True
        self._last_selected_node_id = last_selected_node_id
        restore_candidate_payloads(
            tree=tree,
            payload=payload,
            nodes_by_id=nodes_by_id,
            node_state_by_id=self._node_state_by_id,
            candidate_heap=self._candidate_heap,
            candidate_value_or_none=self._candidate_value_or_none,
            candidate_signature=lambda node, value: self._candidate_signature(
                node=node,
                candidate_value=value,
            ),
        )
        return True

    def choose_node_and_branch_to_open(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: tree_man.TreeExpansions[NodeT],
    ) -> OpeningInstructions[NodeT]:
        """Choose one unopened node according to the Linoo policy."""
        self.latest_selection_report = None
        total_started_at = perf_counter()

        objective = self._require_single_agent_objective(tree)
        collect_started_at = perf_counter()
        with self._diagnostic_phase("select.collect"):
            state_rebuilt, nodes_incrementally_updated = self._ensure_runtime_state(
                tree=tree,
                objective=objective,
                latest_tree_expansions=latest_tree_expansions,
            )
        collect_frontier_state_s = perf_counter() - collect_started_at
        total_nodes_scanned = nodes_incrementally_updated
        if state_rebuilt:
            total_nodes_scanned = sum(
                depth_state.total_nodes
                for depth_state in self._depth_stats_by_depth.values()
            )
        frontier_nodes_scanned = sum(
            depth_state.frontier_count
            for depth_state in self._depth_stats_by_depth.values()
        )
        uncached_terminal_candidates = sum(
            depth_state.uncached_terminal_candidates
            for depth_state in self._depth_stats_by_depth.values()
        )

        if not self._active_frontier_depths():
            raise _no_frontier_nodes_error()

        nodes_by_id = self._nodes_by_id_from_tree(tree)

        choose_depth_started_at = perf_counter()
        with self._diagnostic_phase("select.choose_depth"):
            selected_depth = self._choose_depth_from_cache()
        choose_depth_s = perf_counter() - choose_depth_started_at
        heap_update_started_at = perf_counter()
        with self._diagnostic_phase("select.heap_update"):
            heap_update_diagnostics = self._candidate_heap.make_update_diagnostics()
            selected_depth_frontier_nodes = self._frontier_nodes_at_depth(
                tree=tree,
                nodes_by_id=nodes_by_id,
                depth=selected_depth,
            )
            heap_update_diagnostics.frontier_node_count_seen = len(
                selected_depth_frontier_nodes
            )
            heap_candidates_registered = self._register_frontier_candidates(
                depth=selected_depth,
                nodes=selected_depth_frontier_nodes,
                objective=objective,
                diagnostics=heap_update_diagnostics,
            )
        heap_update_s = perf_counter() - heap_update_started_at
        choose_node_started_at = perf_counter()
        with self._diagnostic_phase("select.choose_node"):
            (
                selected_node,
                stale_candidates_skipped,
                selected_node_rank,
                ranked_candidate_count,
                selected_node_priority,
            ) = self._choose_zipf_node_at_depth(
                tree=tree,
                nodes_by_id=nodes_by_id,
                depth=selected_depth,
                diagnostics=heap_update_diagnostics,
            )
        choose_node_s = perf_counter() - choose_node_started_at
        selected_depth_frontier_count = self._depth_stats_by_depth[
            selected_depth
        ].frontier_count
        make_report_started_at = perf_counter()
        with self._diagnostic_phase("select.report"):
            self.latest_selection_report = self._make_selection_report(
                depth_stats_by_depth=self._depth_stats_by_depth,
                selected_depth=selected_depth,
                selected_node=selected_node,
                collect_frontier_state_s=collect_frontier_state_s,
                choose_depth_s=choose_depth_s,
                heap_update_s=heap_update_s,
                choose_node_s=choose_node_s,
                total_s=perf_counter() - total_started_at,
                total_nodes_scanned=total_nodes_scanned,
                frontier_nodes_scanned=frontier_nodes_scanned,
                uncached_terminal_candidates=uncached_terminal_candidates,
                selected_depth_frontier_count=selected_depth_frontier_count,
                stale_candidates_skipped=stale_candidates_skipped,
                heap_candidates_registered=heap_candidates_registered,
                heap_update_diagnostics=heap_update_diagnostics,
                selected_node_priority=selected_node_priority,
                selected_node_rank=selected_node_rank,
                ranked_candidate_count=ranked_candidate_count,
                state_rebuilt=state_rebuilt,
                nodes_incrementally_updated=nodes_incrementally_updated,
            )
        self.latest_selection_report = replace(
            self.latest_selection_report,
            make_report_s=perf_counter() - make_report_started_at,
            total_s=perf_counter() - total_started_at,
        )

        all_branches_to_open = self.opening_instructor.all_branches_to_open(
            node_to_open=selected_node
        )
        self._last_selected_node_id = selected_node.id
        return create_instructions_to_open_all_branches(
            branches_to_play=all_branches_to_open,
            node_to_open=selected_node,
        )

    def _require_single_agent_objective(
        self,
        tree: trees.Tree[NodeT],
    ) -> SingleAgentMaxObjective[Any]:
        """Return the configured single-agent objective or raise a clear error."""
        try:
            objective = tree.root_node.tree_evaluation.required_objective
        except Exception as exc:
            raise _missing_root_objective_error() from exc

        if not isinstance(objective, SingleAgentMaxObjective):
            raise _incompatible_objective_error(objective)
        return objective

    def _ensure_runtime_state(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
        latest_tree_expansions: tree_man.TreeExpansions[NodeT],
    ) -> tuple[bool, int]:
        """Build or incrementally refresh cached Linoo tree state."""
        if not self._cache_matches(tree=tree, objective=objective):
            self._rebuild_runtime_state(tree=tree, objective=objective)
            return True, 0

        incremental_nodes = self._incremental_nodes_from_expansions(
            tree,
            latest_tree_expansions,
        )
        if incremental_nodes is None:
            self._rebuild_runtime_state(tree=tree, objective=objective)
            return True, 0
        nodes_to_update, created_node_ids = incremental_nodes

        nodes_updated = self._update_runtime_state_incrementally(
            tree=tree,
            objective=objective,
            nodes_to_update=nodes_to_update,
            created_node_ids=created_node_ids,
        )
        tree_node_count = self._tree_node_count(tree)
        if tree_node_count is not None and tree_node_count != self._cached_node_count():
            self._rebuild_runtime_state(tree=tree, objective=objective)
            return True, 0
        return False, nodes_updated

    def _cache_matches(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
    ) -> bool:
        """Return whether cached state belongs to this tree/objective pair."""
        if not self._cache_initialized:
            return False
        tree_identity = id(tree.root_node)
        objective_identity = id(objective)
        return (
            self._cache_tree_identity == tree_identity
            and self._cache_objective_identity == objective_identity
        )

    def _clear_runtime_state(self) -> None:
        """Clear all cached selector runtime state."""
        self._depth_stats_by_depth.clear()
        self._frontier_node_ids_by_depth.clear()
        self._node_state_by_id.clear()
        self._candidate_heap.clear()
        self._cache_initialized = False
        self._cache_tree_identity = None
        self._cache_objective_identity = None
        self._last_selected_node_id = None

    def _node_state_or_none(self, node_id: int) -> _LinooNodeState | None:
        """Return existing Linoo state without materializing defaults."""
        return self._node_state_by_id.get(node_id)

    def _node_state(
        self,
        node_id: int,
        *,
        depth: int,
    ) -> _LinooNodeState:
        """Return Linoo state, materializing default state for mutation."""
        state = self._node_state_or_none(node_id)
        if state is not None:
            return state
        return _LinooNodeState(
            node_id=node_id,
            depth=depth,
            status=_LINOO_DEFAULT_NODE_STATUS,
        )

    def _set_node_state_if_non_default(
        self,
        node_id: int,
        state: _LinooNodeState,
    ) -> None:
        """Store state only when non-default; otherwise remove the entry."""
        if state.is_default():
            self._node_state_by_id.pop(node_id, None)
            return
        self._node_state_by_id[node_id] = state

    def _discard_default_node_state(self, node_id: int) -> None:
        """Remove state if it is default-equivalent."""
        state = self._node_state_or_none(node_id)
        if state is not None and state.is_default():
            self._node_state_by_id.pop(node_id, None)

    def _cached_node_count(self) -> int:
        """Return the number of nodes represented by aggregate depth stats."""
        return sum(stats.total_nodes for stats in self._depth_stats_by_depth.values())

    def _nodes_by_id_from_tree(self, tree: trees.Tree[NodeT]) -> dict[int, NodeT]:
        """Return live tree nodes keyed by public node id."""
        return {
            node.id: node
            for absolute_tree_depth in tree.descendants
            for node in tree.descendants[absolute_tree_depth].values()
        }

    def _rebuild_runtime_state(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
    ) -> int:
        """Reconstruct Linoo state by scanning the whole tree once."""
        self._clear_runtime_state()
        tree_identity = id(tree.root_node)
        objective_identity = id(objective)
        self._cache_tree_identity = tree_identity
        self._cache_objective_identity = objective_identity
        scanned_count = 0

        absolute_tree_depth: int
        for absolute_tree_depth in tree.descendants:
            node: NodeT
            for node in tree.descendants[absolute_tree_depth].values():
                self._add_node_state(
                    node=node,
                    depth=tree.node_depth(node),
                    status=self._classify_node(node),
                )
                scanned_count += 1

        self._cache_initialized = True
        return scanned_count

    def _update_runtime_state_incrementally(
        self,
        *,
        tree: trees.Tree[NodeT],
        objective: SingleAgentMaxObjective[Any],
        nodes_to_update: dict[int, NodeT],
        created_node_ids: set[int],
    ) -> int:
        """Refresh cached classifications for one expansion wave."""
        self._cache_tree_identity = id(tree.root_node)
        self._cache_objective_identity = id(objective)
        tree_node_count = self._tree_node_count(tree)
        cache_already_represents_current_tree = (
            tree_node_count is not None and tree_node_count == self._cached_node_count()
        )
        for node_id, node in nodes_to_update.items():
            self._reclassify_node(
                tree=tree,
                node=node,
                already_cached=(
                    cache_already_represents_current_tree
                    or node_id not in created_node_ids
                ),
            )
        return len(nodes_to_update)

    def _incremental_nodes_from_expansions(
        self,
        tree: trees.Tree[NodeT],
        latest_tree_expansions: tree_man.TreeExpansions[NodeT],
    ) -> tuple[dict[int, NodeT], set[int]] | None:
        """Collect touched nodes using the explicit ``TreeExpansions`` API."""
        try:
            creation_expansions = latest_tree_expansions.expansions_with_node_creation
            connection_expansions = (
                latest_tree_expansions.expansions_without_node_creation
            )
            created_nodes = latest_tree_expansions.created_nodes
            affected_child_nodes = latest_tree_expansions.affected_child_nodes
        except AttributeError:
            # Defensive fallback for legacy tests/non-standard selector callers.
            return None
        nodes_by_id: dict[int, NodeT] = {}
        if self._last_selected_node_id is not None:
            tree_nodes_by_id = self._nodes_by_id_from_tree(tree)
            last_selected_node = tree_nodes_by_id.get(self._last_selected_node_id)
            if last_selected_node is not None:
                nodes_by_id[self._last_selected_node_id] = last_selected_node

        created_node_ids: set[int] = set()
        for expansion in (*creation_expansions, *connection_expansions):
            parent_node = expansion.parent_node
            if parent_node is not None:
                nodes_by_id[parent_node.id] = parent_node
            child_node = expansion.child_node
            nodes_by_id[child_node.id] = child_node
        for expansion in creation_expansions:
            created_node_ids.add(expansion.child_node.id)

        for node in created_nodes():
            nodes_by_id[node.id] = node
            created_node_ids.add(node.id)
        for node in affected_child_nodes():
            nodes_by_id[node.id] = node
        return nodes_by_id, created_node_ids

    def _reclassify_node(
        self,
        *,
        tree: trees.Tree[NodeT],
        node: NodeT,
        already_cached: bool,
    ) -> None:
        """Refresh cached state for one node that may have changed."""
        depth = tree.node_depth(node)
        if already_cached:
            self._remove_node_state(node_id=node.id, depth=depth)
        self._add_node_state(
            node=node,
            depth=depth,
            status=self._classify_node(node),
        )

    def _add_node_state(
        self,
        *,
        node: NodeT,
        depth: int,
        status: _LinooNodeStatus,
    ) -> None:
        """Add one node classification to cached counts."""
        depth_stats = self._depth_stats_by_depth.setdefault(depth, _LinooDepthStats())
        depth_stats.total_nodes += 1
        depth_stats.increment(status)
        if status == "frontier":
            self._frontier_node_ids_by_depth.setdefault(depth, set()).add(node.id)
        self._set_node_state_if_non_default(
            node.id,
            _LinooNodeState(
                node_id=node.id,
                depth=depth,
                status=status,
            ),
        )

    def _remove_node_state(self, *, node_id: int, depth: int) -> None:
        """Remove one cached node classification if it already exists."""
        old_state = self._node_state(node_id, depth=depth)
        self._node_state_by_id.pop(node_id, None)
        depth_stats = self._depth_stats_by_depth.get(old_state.depth)
        if depth_stats is None:
            return
        depth_stats.total_nodes -= 1
        depth_stats.decrement(old_state.status)
        if old_state.status == "frontier":
            frontier_ids = self._frontier_node_ids_by_depth.get(old_state.depth)
            if frontier_ids is not None:
                frontier_ids.discard(node_id)
                if not frontier_ids:
                    self._frontier_node_ids_by_depth.pop(old_state.depth, None)
            self._mark_candidate_stale(node_id)
        if depth_stats.empty():
            self._depth_stats_by_depth.pop(old_state.depth, None)

    def _classify_node(self, node: NodeT) -> _LinooNodeStatus:
        """Return the current cached Linoo status for one node."""
        if self._is_terminal_node(node):
            return "terminal"
        if node.all_branches_generated:
            return "opened"
        if node.tree_evaluation.has_exact_value():
            return "exact"
        if self._candidate_value_or_none(node) is None:
            return "uncached_terminal_candidate"
        return "frontier"

    def _mark_candidate_stale(self, node_id: int) -> None:
        """Invalidate existing heap entries for one node lazily."""
        self._candidate_heap.mark_stale(node_id)

    def _tree_node_count(self, tree: trees.Tree[NodeT]) -> int | None:
        """Return the tree's known node count, when exposed by the tree object."""
        nodes_count = getattr(tree, "nodes_count", None)
        return nodes_count if isinstance(nodes_count, int) else None

    def _active_frontier_depths(self) -> tuple[int, ...]:
        """Return depths that currently have at least one frontier node."""
        return tuple(
            sorted(
                depth
                for depth, frontier_ids in self._frontier_node_ids_by_depth.items()
                if frontier_ids
            )
        )

    def _choose_depth_from_cache(self) -> int:
        """Choose the active depth using the configured Linoo depth policy."""
        return choose_depth(
            depth_selection_policy=self.depth_selection_policy,
            depth_stats_by_depth=self._depth_stats_by_depth,
            active_depths=self._active_frontier_depths(),
            random_generator=self.random_generator,
        )

    def _frontier_nodes_at_depth(
        self,
        *,
        tree: trees.Tree[NodeT],
        nodes_by_id: dict[int, NodeT],
        depth: int,
    ) -> list[NodeT]:
        """Return cached frontier nodes at one depth."""
        frontier_nodes: list[NodeT] = []
        for node_id in self._frontier_node_ids_by_depth.get(depth, set()):
            node = nodes_by_id.get(node_id)
            if node is not None and tree.node_depth(node) == depth:
                frontier_nodes.append(node)
        return frontier_nodes

    def _register_frontier_candidates(
        self,
        *,
        depth: int,
        nodes: list[NodeT],
        objective: SingleAgentMaxObjective[Any],
        diagnostics: _LinooHeapUpdateDiagnostics,
    ) -> int:
        """Update cached heap entries for the selected depth only."""
        return self._candidate_heap.register_frontier_candidates(
            depth=depth,
            nodes=nodes,
            objective=objective,
            diagnostics=diagnostics,
            require_candidate_value=self._require_candidate_value,
            candidate_signature=lambda node, value: self._candidate_signature(
                node=node,
                candidate_value=value,
            ),
            candidate_priority=lambda objective_, value, node, diagnostics_: (
                self._candidate_priority(
                    objective=objective_,
                    value=value,
                    node=node,
                    diagnostics=diagnostics_,
                )
            ),
            record_candidate_source=lambda source, diagnostics_: (
                self._record_candidate_source(
                    source=source,
                    diagnostics=diagnostics_,
                )
            ),
            diagnostic_phase=self._diagnostic_phase,
        )

    def _register_frontier_candidate(
        self,
        *,
        depth: int,
        node: NodeT,
        objective: SingleAgentMaxObjective[Any],
        diagnostics: _LinooHeapUpdateDiagnostics,
    ) -> bool:
        """Push a heap entry when one node is new, changed, or reactivated."""
        return self._candidate_heap.register_frontier_candidate(
            depth=depth,
            node=node,
            objective=objective,
            diagnostics=diagnostics,
            require_candidate_value=self._require_candidate_value,
            candidate_signature=lambda node, value: self._candidate_signature(
                node=node,
                candidate_value=value,
            ),
            candidate_priority=lambda objective_, value, node, diagnostics_: (
                self._candidate_priority(
                    objective=objective_,
                    value=value,
                    node=node,
                    diagnostics=diagnostics_,
                )
            ),
            record_candidate_source=lambda source, diagnostics_: (
                self._record_candidate_source(
                    source=source,
                    diagnostics=diagnostics_,
                )
            ),
            diagnostic_phase=self._diagnostic_phase,
        )

    def _rank_valid_candidates_at_depth(
        self,
        *,
        tree: trees.Tree[NodeT],
        nodes_by_id: dict[int, NodeT],
        depth: int,
        diagnostics: _LinooHeapUpdateDiagnostics | None = None,
    ) -> tuple[list[_LinooRankedCandidate], int]:
        """Return valid candidates sorted by priority, compacting stale heap entries."""
        return self._candidate_heap.rank_valid_candidates_at_depth(
            depth=depth,
            frontier_node_ids=self._frontier_node_ids_by_depth.get(depth, set()),
            is_current_frontier_candidate=lambda node_id, expected_depth: (
                self._is_current_frontier_candidate(
                    tree=tree,
                    nodes_by_id=nodes_by_id,
                    node_id=node_id,
                    expected_depth=expected_depth,
                )
            ),
            diagnostic_phase=self._diagnostic_phase,
            diagnostics=diagnostics,
        )

    def _choose_zipf_node_at_depth(
        self,
        *,
        tree: trees.Tree[NodeT],
        nodes_by_id: dict[int, NodeT],
        depth: int,
        diagnostics: _LinooHeapUpdateDiagnostics,
    ) -> tuple[NodeT, int, int, int, float]:
        """Sample one valid frontier candidate using Zipf rank weights."""
        ranked_candidates, stale_candidates_skipped = (
            self._rank_valid_candidates_at_depth(
                tree=tree,
                nodes_by_id=nodes_by_id,
                depth=depth,
                diagnostics=diagnostics,
            )
        )
        if not ranked_candidates:
            raise _no_frontier_nodes_error()

        total_weight = sum(1.0 / rank for rank in range(1, len(ranked_candidates) + 1))
        threshold = self.random_generator.random() * total_weight
        cumulative_weight = 0.0
        for rank, candidate in enumerate(ranked_candidates, start=1):
            cumulative_weight += 1.0 / rank
            if threshold < cumulative_weight or rank == len(ranked_candidates):
                priority_key, node_id, _version = candidate
                node = nodes_by_id.get(node_id)
                if node is not None and tree.node_depth(node) != depth:
                    node = None
                if node is None:
                    raise _no_frontier_nodes_error()
                return (
                    node,
                    stale_candidates_skipped,
                    rank,
                    len(ranked_candidates),
                    -priority_key,
                )

        raise _no_frontier_nodes_error()

    def _is_current_frontier_candidate(
        self,
        *,
        tree: trees.Tree[NodeT],
        nodes_by_id: dict[int, NodeT],
        node_id: int,
        expected_depth: int,
    ) -> bool:
        """Return whether one cached node is still selectable frontier."""
        node_state = self._node_state_or_none(node_id)
        node = nodes_by_id.get(node_id)
        return (
            node_state is not None
            and node is not None
            and node_state.depth == expected_depth
            and node_state.status == "frontier"
            and tree.node_depth(node) == expected_depth
        )

    def _make_selection_report(
        self,
        *,
        depth_stats_by_depth: dict[int, _LinooDepthStats],
        selected_depth: int,
        selected_node: NodeT,
        collect_frontier_state_s: float,
        choose_depth_s: float,
        heap_update_s: float,
        choose_node_s: float,
        total_s: float,
        total_nodes_scanned: int,
        frontier_nodes_scanned: int,
        uncached_terminal_candidates: int,
        selected_depth_frontier_count: int,
        stale_candidates_skipped: int,
        heap_candidates_registered: int,
        heap_update_diagnostics: _LinooHeapUpdateDiagnostics,
        selected_node_priority: float,
        selected_node_rank: int,
        ranked_candidate_count: int,
        state_rebuilt: bool,
        nodes_incrementally_updated: int,
    ) -> LinooSelectionReport:
        """Build the structured latest-selection table without affecting policy."""
        rows: list[LinooDepthSelectionRow] = []
        selected_depth_selection_index: int | None = None
        selected_depth_selection_weight: float | None = None
        selected_depth_selection_probability: float | None = None
        active_depths = tuple(
            depth
            for depth, depth_state in depth_stats_by_depth.items()
            if depth_state.frontier_count > 0
        )
        total_inverse_depth_weight = sum(
            inverse_depth_weight(depth) for depth in active_depths
        )

        for depth, depth_state in depth_stats_by_depth.items():
            opened_count = depth_state.opened_count
            active = depth_state.frontier_count > 0
            selected = depth == selected_depth
            selection_index = opened_count * (depth + 1) if active else None
            selection_weight: float | None = None
            selection_probability: float | None = None
            if (
                active
                and self.depth_selection_policy == "inverse_depth"
                and total_inverse_depth_weight > 0.0
            ):
                selection_weight = inverse_depth_weight(depth)
                selection_probability = selection_weight / total_inverse_depth_weight
            row = LinooDepthSelectionRow(
                depth=depth,
                total_nodes=depth_state.total_nodes,
                opened_count=opened_count,
                frontier_count=depth_state.frontier_count,
                terminal_count=depth_state.terminal_count,
                exact_count=depth_state.exact_count,
                uncached_terminal_candidates=(depth_state.uncached_terminal_candidates),
                non_openable_count=depth_state.non_openable_count,
                selection_index=selection_index,
                active=active,
                selected=selected,
                selection_weight=selection_weight,
                selection_probability=selection_probability,
            )
            rows.append(row)
            if selected:
                selected_depth_selection_index = selection_index
                selected_depth_selection_weight = selection_weight
                selected_depth_selection_probability = selection_probability

        sorted_rows = tuple(
            sorted(
                rows,
                key=lambda row: depth_selection_row_sort_key(
                    depth_selection_policy=self.depth_selection_policy,
                    row=row,
                ),
            )
        )
        if selected_depth_selection_index is None:
            selected_depth_selection_index = depth_stats_by_depth[
                selected_depth
            ].opened_count * (selected_depth + 1)

        return LinooSelectionReport(
            selected_depth=selected_depth,
            selected_node_id=selected_node.id,
            selected_node_direct_value=self._direct_value_score_or_none(selected_node),
            selected_node_candidate_value=(
                cast("Value", self._require_candidate_value(selected_node).value).score
            ),
            selected_node_priority=selected_node_priority,
            selected_node_rank=selected_node_rank,
            ranked_candidate_count=ranked_candidate_count,
            node_selection_policy="zipf_rank",
            depth_selection_policy=self.depth_selection_policy,
            selected_depth_selection_index=selected_depth_selection_index,
            depth_rows=sorted_rows,
            selected_depth_selection_weight=selected_depth_selection_weight,
            selected_depth_selection_probability=(selected_depth_selection_probability),
            collect_frontier_state_s=collect_frontier_state_s,
            choose_depth_s=choose_depth_s,
            heap_update_s=heap_update_s,
            choose_node_s=choose_node_s,
            make_report_s=None,
            total_s=total_s,
            depth_row_count=len(sorted_rows),
            total_nodes_scanned=total_nodes_scanned,
            frontier_nodes_scanned=frontier_nodes_scanned,
            uncached_terminal_candidates=uncached_terminal_candidates,
            selected_depth_frontier_count=selected_depth_frontier_count,
            stale_candidates_skipped=stale_candidates_skipped,
            heap_candidates_registered=heap_candidates_registered,
            heap_update_candidate_count=heap_update_diagnostics.candidate_count,
            heap_update_push_count=heap_update_diagnostics.push_count,
            heap_update_pop_count=heap_update_diagnostics.pop_count,
            heap_update_stale_skip_count=heap_update_diagnostics.stale_skip_count,
            heap_update_signature_check_count=(
                heap_update_diagnostics.signature_check_count
            ),
            heap_update_signature_recompute_count=(
                heap_update_diagnostics.signature_recompute_count
            ),
            heap_update_version_mismatch_count=(
                heap_update_diagnostics.version_mismatch_count
            ),
            heap_update_priority_state_free_count=(
                heap_update_diagnostics.priority_state_free_count
            ),
            heap_update_priority_stateful_fallback_count=(
                heap_update_diagnostics.priority_stateful_fallback_count
            ),
            heap_update_candidate_direct_count=(
                heap_update_diagnostics.candidate_source_counts.get(
                    ValueCandidateSource.DIRECT_SELF,
                    0,
                )
            ),
            heap_update_candidate_tree_count=(
                heap_update_diagnostics.candidate_source_counts.get(
                    ValueCandidateSource.TREE_CHILD,
                    0,
                )
            ),
            heap_update_candidate_unknown_count=(
                heap_update_diagnostics.candidate_source_counts.get(
                    ValueCandidateSource.NONE,
                    0,
                )
            ),
            heap_update_total_heap_entries=(heap_update_diagnostics.total_heap_entries),
            heap_update_max_heap_size=heap_update_diagnostics.max_heap_size,
            heap_update_depth_count=heap_update_diagnostics.depth_count,
            heap_update_frontier_node_count_seen=(
                heap_update_diagnostics.frontier_node_count_seen
            ),
            state_rebuilt=state_rebuilt,
            nodes_incrementally_updated=nodes_incrementally_updated,
        )

    def _is_terminal_node(self, node: NodeT) -> bool:
        """Return cached terminality only; Linoo must not recompute game-over."""
        is_terminal = getattr(node.tree_evaluation, "is_terminal", None)
        return bool(is_terminal()) if callable(is_terminal) else False

    def _direct_value_or_none(self, node: NodeT) -> Value | None:
        """Return direct value access for openable-candidate classification."""
        try:
            return node.tree_evaluation.direct_value
        except AttributeError as exc:
            raise _missing_direct_value_attribute_error(node.id) from exc

    def _require_direct_value(self, node: NodeT) -> Value:
        """Return one node's direct value or raise a clear Linoo error."""
        direct_value = self._direct_value_or_none(node)

        if direct_value is None:
            raise _missing_direct_value_error(node.id)
        return direct_value

    def _direct_value_score_or_none(self, node: NodeT) -> float | None:
        """Return direct score for observability without making it required."""
        direct_value = self._direct_value_or_none(node)
        return None if direct_value is None else direct_value.score

    def _candidate_value_or_none(self, node: NodeT) -> Value | None:
        """Return the effective search-facing value for one Linoo candidate."""
        return self._candidate_value_candidate(node).value

    def _require_candidate_value(self, node: NodeT) -> ValueCandidate:
        """Return one node's effective candidate value or raise a clear error."""
        candidate = self._candidate_value_candidate(node)
        if candidate.value is None:
            raise _missing_direct_value_error(node.id)
        return candidate

    def _candidate_value_candidate(self, node: NodeT) -> ValueCandidate:
        """Return the source-aware effective candidate, falling back to direct."""
        get_effective_value_candidate = getattr(
            node.tree_evaluation,
            "get_effective_value_candidate",
            None,
        )
        if callable(get_effective_value_candidate):
            candidate = get_effective_value_candidate()
            if isinstance(candidate, ValueCandidate):
                return candidate
            value = getattr(candidate, "value", None)
            source = getattr(candidate, "source", ValueCandidateSource.NONE)
            if isinstance(source, str):
                try:
                    source = ValueCandidateSource(source)
                except ValueError:
                    source = ValueCandidateSource.NONE
            if value is not None:
                return ValueCandidate(value=value, source=source)
            return ValueCandidate.none()

        direct_value = self._direct_value_or_none(node)
        if direct_value is None:
            return ValueCandidate.none()
        return ValueCandidate.direct(direct_value)

    def _candidate_priority(
        self,
        *,
        objective: SingleAgentMaxObjective[Any],
        value: Value,
        node: NodeT,
        diagnostics: _LinooHeapUpdateDiagnostics,
    ) -> float:
        """Project candidate value using state-free objective hooks when available."""
        state_free_evaluator = getattr(objective, "evaluate_value_without_state", None)
        if callable(state_free_evaluator):
            diagnostics.priority_state_free_count += 1
            return cast("float", state_free_evaluator(value))

        diagnostics.priority_stateful_fallback_count += 1
        return objective.evaluate_value(value, node.state)

    def _record_candidate_source(
        self,
        *,
        source: ValueCandidateSource,
        diagnostics: _LinooHeapUpdateDiagnostics,
    ) -> None:
        """Count value provenance for heap-push diagnostics."""
        diagnostics.candidate_source_counts[source] = (
            diagnostics.candidate_source_counts.get(source, 0) + 1
        )

    def _candidate_signature(self, *, node: NodeT, candidate_value: Value) -> object:
        """Return the cheap invalidation signature for one heap candidate."""
        return (
            id(node),
            candidate_value,
            node.all_branches_generated,
            node.tree_evaluation.has_exact_value(),
            self._is_terminal_node(node),
        )
