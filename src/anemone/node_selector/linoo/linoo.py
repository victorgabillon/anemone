"""Single-player depth-aware selector based on direct node values."""

from __future__ import annotations

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from anemone.node_selector.opening_instructions import (
    OpeningInstructions,
    OpeningInstructor,
    create_instructions_to_open_all_branches,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.objectives import SingleAgentMaxObjective

if TYPE_CHECKING:
    from valanga.evaluations import Value

    from anemone import tree_manager as tree_man
    from anemone import trees
    from anemone.checkpoints.payloads import (
        LinooCandidatesByDepthCheckpointPayload,
        LinooDepthStatsCheckpointPayload,
        LinooNodeStateCheckpointPayload,
        LinooSelectorCheckpointPayload,
    )
    from anemone.node_selector.node_selector_types import NodeSelectorType


@dataclass
class LinooArgs:
    """Arguments for the Linoo node selector."""

    type: Literal[NodeSelectorType.LINOO]


@dataclass(frozen=True, slots=True)
class LinooDepthSelectionRow:
    """Selector-observable summary for one depth in the live tree."""

    depth: int
    total_nodes: int
    opened_count: int
    frontier_count: int
    terminal_count: int
    exact_count: int
    uncached_terminal_candidates: int
    non_openable_count: int
    selection_index: int | None
    active: bool
    selected: bool

    def __post_init__(self) -> None:
        """Keep the diagnostic accounting honest in debug runs."""
        assert self.total_nodes == (
            self.opened_count
            + self.frontier_count
            + self.terminal_count
            + self.exact_count
            + self.uncached_terminal_candidates
            + self.non_openable_count
        )


@dataclass(frozen=True, slots=True)
class LinooSelectionReport:
    """Structured observability payload for the latest Linoo selection."""

    selected_depth: int
    selected_node_id: int
    selected_node_direct_value: float | None
    selected_depth_selection_index: int
    depth_rows: tuple[LinooDepthSelectionRow, ...]
    collect_frontier_state_s: float | None = None
    choose_depth_s: float | None = None
    heap_update_s: float | None = None
    choose_node_s: float | None = None
    make_report_s: float | None = None
    total_s: float | None = None
    depth_row_count: int | None = None
    total_nodes_scanned: int | None = None
    frontier_nodes_scanned: int | None = None
    uncached_terminal_candidates: int | None = None
    selected_depth_frontier_count: int | None = None
    stale_candidates_skipped: int | None = None
    heap_candidates_registered: int | None = None
    state_rebuilt: bool | None = None
    nodes_incrementally_updated: int | None = None

    def format_depth_table(self) -> str:
        """Return an aligned text table for Linoo depth diagnostics."""
        headers = (
            "depth",
            "total",
            "opened",
            "frontier",
            "terminal",
            "exact",
            "uncached_terminal",
            "non_openable",
            "index",
            "selected",
        )
        rows = [
            (
                str(row.depth),
                str(row.total_nodes),
                str(row.opened_count),
                str(row.frontier_count),
                str(row.terminal_count),
                str(row.exact_count),
                str(row.uncached_terminal_candidates),
                str(row.non_openable_count),
                _format_optional_int(row.selection_index),
                "yes" if row.selected else "no",
            )
            for row in self.depth_rows
        ]
        return _format_table(headers, rows)


type _LinooNodeStatus = Literal[
    "opened",
    "frontier",
    "terminal",
    "exact",
    "uncached_terminal_candidate",
    "non_openable",
]


@dataclass(init=False, slots=True)
class _LinooDepthStats:
    """Mutable accounting bucket for one depth in the cached tree view."""

    total_nodes: int
    opened_count: int
    frontier_count: int
    terminal_count: int
    exact_count: int
    uncached_terminal_candidates: int
    non_openable_count: int

    def __init__(self) -> None:
        """Initialize empty accounting for one depth."""
        self.total_nodes = 0
        self.opened_count = 0
        self.frontier_count = 0
        self.terminal_count = 0
        self.exact_count = 0
        self.uncached_terminal_candidates = 0
        self.non_openable_count = 0

    def count_for(self, status: _LinooNodeStatus) -> int:
        """Return the counter value for one cached node status."""
        return getattr(self, _LINOO_NODE_STATUS_COUNTERS[status])

    def increment(self, status: _LinooNodeStatus) -> None:
        """Add one node with ``status`` to this depth."""
        setattr(
            self,
            _LINOO_NODE_STATUS_COUNTERS[status],
            self.count_for(status) + 1,
        )

    def decrement(self, status: _LinooNodeStatus) -> None:
        """Remove one node with ``status`` from this depth."""
        next_count = self.count_for(status) - 1
        assert next_count >= 0
        setattr(self, _LINOO_NODE_STATUS_COUNTERS[status], next_count)

    def empty(self) -> bool:
        """Return whether this depth no longer tracks any nodes."""
        return self.total_nodes == 0


_LINOO_NODE_STATUS_COUNTERS: dict[_LinooNodeStatus, str] = {
    "opened": "opened_count",
    "frontier": "frontier_count",
    "terminal": "terminal_count",
    "exact": "exact_count",
    "uncached_terminal_candidate": "uncached_terminal_candidates",
    "non_openable": "non_openable_count",
}
_LINOO_NODE_STATUSES = frozenset(_LINOO_NODE_STATUS_COUNTERS)


@dataclass(frozen=True, slots=True)
class _LinooNodeState[NodeT]:
    """Cached Linoo classification for one tree node."""

    node: NodeT
    depth: int
    status: _LinooNodeStatus


type _LinooHeapEntry[NodeT] = tuple[float, int, int, NodeT]


class LinooSelectionError(ValueError):
    """Base error for invalid Linoo selection contexts."""


class LinooDirectValueUnavailableError(LinooSelectionError):
    """Raised when a frontier node does not expose a direct single-player value."""


class LinooIncompatibleObjectiveError(LinooSelectionError):
    """Raised when Linoo is used outside the single-player max setting."""


def _no_frontier_nodes_error() -> LinooSelectionError:
    return LinooSelectionError(
        "Linoo could not find any unopened unresolved nodes in the current tree."
    )


def _missing_root_objective_error() -> LinooIncompatibleObjectiveError:
    return LinooIncompatibleObjectiveError(
        "Linoo requires a configured SingleAgentMaxObjective on the tree root."
    )


def _incompatible_objective_error(objective: object) -> LinooIncompatibleObjectiveError:
    return LinooIncompatibleObjectiveError(
        f"Linoo requires a SingleAgentMaxObjective; got {type(objective).__name__}."
    )


def _missing_direct_value_attribute_error(
    node_id: int,
) -> LinooDirectValueUnavailableError:
    return LinooDirectValueUnavailableError(
        "Linoo requires a direct single-player value on every frontier node; "
        f"node {node_id} does not expose tree_evaluation.direct_value."
    )


def _missing_direct_value_error(node_id: int) -> LinooDirectValueUnavailableError:
    return LinooDirectValueUnavailableError(
        "Linoo requires a direct single-player value on every frontier node; "
        f"node {node_id} has no direct_value."
    )


def _invalid_linoo_checkpoint_payload_error() -> ValueError:
    """Return the generic error for rejected optional selector state."""
    return ValueError("Invalid Linoo selector checkpoint payload.")


def _format_optional_int(value: int | None) -> str:
    """Format an optional integer for compact table display."""
    return "-" if value is None else str(value)


def _format_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> str:
    """Format string rows as a simple aligned whitespace table."""
    widths = [
        max([len(headers[column]), *(len(row[column]) for row in rows)])
        for column in range(len(headers))
    ]
    formatted_rows = [
        " ".join(
            value.rjust(widths[column])
            for column, value in enumerate(row)
        )
        for row in rows
    ]
    return "\n".join(
        [
            " ".join(
                header.rjust(widths[column])
                for column, header in enumerate(headers)
            ),
            *formatted_rows,
        ]
    )


class Linoo[NodeT: AlgorithmNode[Any] = AlgorithmNode[Any]]:
    """Depth-aware single-player selector using frontier direct values.

    Linoo first selects the frontier depth minimizing
    ``opened_count(depth) * (depth + 1)``, then expands the unopened node at
    that depth with the best direct value.

    PR1 incremental runtime cache assumption: direct values of existing frontier
    nodes do not change between selections unless the selector cache is
    explicitly invalidated or the node is part of the latest expansion wave.
    """

    opening_instructor: OpeningInstructor
    latest_selection_report: LinooSelectionReport | None
    _depth_stats_by_depth: dict[int, _LinooDepthStats]
    _frontier_node_ids_by_depth: dict[int, set[int]]
    _node_state_by_id: dict[int, _LinooNodeState[NodeT]]
    _candidates_by_depth: dict[int, list[_LinooHeapEntry[NodeT]]]
    _candidate_versions_by_node_id: dict[int, int]
    _candidate_signature_by_node_id: dict[int, object]
    _candidate_heap_present_by_node_id: dict[int, bool]
    _cache_tree_identity: int | None
    _cache_objective_identity: int | None
    _cache_initialized: bool
    _last_selected_node: NodeT | None

    def __init__(self, opening_instructor: OpeningInstructor) -> None:
        """Store the opening instructor used to materialize branch openings."""
        self.opening_instructor = opening_instructor
        self.latest_selection_report = None
        self._depth_stats_by_depth = {}
        self._frontier_node_ids_by_depth = {}
        self._node_state_by_id = {}
        self._candidates_by_depth = {}
        self._candidate_versions_by_node_id = {}
        self._candidate_signature_by_node_id = {}
        self._candidate_heap_present_by_node_id = {}
        self._cache_tree_identity = None
        self._cache_objective_identity = None
        self._cache_initialized = False
        self._last_selected_node = None

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
            depth_stats=self._depth_stats_payload_from_cache(),
            node_states=self._node_states_payload_from_cache(),
            candidates_by_depth=self._candidate_payloads_from_cache(),
            last_selected_node_id=(
                self._last_selected_node.id
                if self._last_selected_node is not None
                else None
            ),
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
            node_states = self._restore_node_states_from_payload(
                tree=tree,
                nodes_by_id=nodes_by_id,
                payload=payload,
            )
            depth_stats = self._depth_stats_from_node_states(node_states)
            if not self._depth_stats_match_payload(depth_stats, payload.depth_stats):
                raise _invalid_linoo_checkpoint_payload_error()
            frontier_node_ids = self._frontier_ids_from_node_states(node_states)
            last_selected_node = (
                None
                if payload.last_selected_node_id is None
                else nodes_by_id[payload.last_selected_node_id]
            )
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
        self._last_selected_node = last_selected_node
        self._restore_candidate_payloads(
            tree=tree,
            payload=payload,
            nodes_by_id=nodes_by_id,
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

        choose_depth_started_at = perf_counter()
        selected_depth = self._choose_depth_from_cache()
        choose_depth_s = perf_counter() - choose_depth_started_at
        heap_update_started_at = perf_counter()
        heap_candidates_registered = self._register_frontier_candidates(
            depth=selected_depth,
            nodes=self._frontier_nodes_at_depth(selected_depth),
            objective=objective,
        )
        heap_update_s = perf_counter() - heap_update_started_at
        choose_node_started_at = perf_counter()
        selected_node, stale_candidates_skipped = self._choose_best_node_at_depth(
            tree=tree,
            depth=selected_depth,
        )
        choose_node_s = perf_counter() - choose_node_started_at
        selected_depth_frontier_count = self._depth_stats_by_depth[
            selected_depth
        ].frontier_count
        make_report_started_at = perf_counter()
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
            state_rebuilt=state_rebuilt,
            nodes_incrementally_updated=nodes_incrementally_updated,
        )
        if self.latest_selection_report is not None:
            self.latest_selection_report = replace(
                self.latest_selection_report,
                make_report_s=perf_counter() - make_report_started_at,
                total_s=perf_counter() - total_started_at,
            )

        all_branches_to_open = self.opening_instructor.all_branches_to_open(
            node_to_open=selected_node
        )
        self._last_selected_node = selected_node
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

        nodes_to_update = self._incremental_nodes_from_expansions(
            latest_tree_expansions
        )
        if nodes_to_update is None:
            self._rebuild_runtime_state(tree=tree, objective=objective)
            return True, 0

        nodes_updated = self._update_runtime_state_incrementally(
            tree=tree,
            objective=objective,
            nodes_to_update=nodes_to_update,
        )
        tree_node_count = self._tree_node_count(tree)
        if tree_node_count is not None and tree_node_count != len(self._node_state_by_id):
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
        self._candidates_by_depth.clear()
        self._candidate_versions_by_node_id.clear()
        self._candidate_signature_by_node_id.clear()
        self._candidate_heap_present_by_node_id.clear()
        self._cache_initialized = False
        self._cache_tree_identity = None
        self._cache_objective_identity = None
        self._last_selected_node = None

    def _nodes_by_id_from_tree(self, tree: trees.Tree[NodeT]) -> dict[int, NodeT]:
        """Return live tree nodes keyed by public node id."""
        return {
            node.id: node
            for absolute_tree_depth in tree.descendants
            for node in tree.descendants[absolute_tree_depth].values()
        }

    def _depth_stats_payload_from_cache(
        self,
    ) -> list[LinooDepthStatsCheckpointPayload]:
        """Serialize cached depth stats in stable depth order."""
        from anemone.checkpoints.payloads import (  # pylint: disable=import-outside-toplevel
            LinooDepthStatsCheckpointPayload,
        )

        return [
            LinooDepthStatsCheckpointPayload(
                depth=depth,
                total_nodes=stats.total_nodes,
                opened_count=stats.opened_count,
                frontier_count=stats.frontier_count,
                terminal_count=stats.terminal_count,
                exact_count=stats.exact_count,
                uncached_terminal_candidates=stats.uncached_terminal_candidates,
                non_openable_count=stats.non_openable_count,
            )
            for depth, stats in sorted(self._depth_stats_by_depth.items())
        ]

    def _node_states_payload_from_cache(
        self,
    ) -> list[LinooNodeStateCheckpointPayload]:
        """Serialize cached node classifications without live node objects."""
        from anemone.checkpoints.payloads import (  # pylint: disable=import-outside-toplevel
            LinooNodeStateCheckpointPayload,
        )

        return [
            LinooNodeStateCheckpointPayload(
                node_id=node_id,
                depth=node_state.depth,
                status=node_state.status,
            )
            for node_id, node_state in sorted(self._node_state_by_id.items())
        ]

    def _candidate_payloads_from_cache(
        self,
    ) -> list[LinooCandidatesByDepthCheckpointPayload]:
        """Serialize non-stale candidate heap entries without mutating heaps."""
        from anemone.checkpoints.payloads import (  # pylint: disable=import-outside-toplevel
            LinooCandidateCheckpointPayload,
            LinooCandidatesByDepthCheckpointPayload,
        )

        payloads: list[LinooCandidatesByDepthCheckpointPayload] = []
        for depth, heap in sorted(self._candidates_by_depth.items()):
            candidates = [
                LinooCandidateCheckpointPayload(
                    node_id=node_id,
                    depth=depth,
                    priority=-priority_key,
                    version=version,
                )
                for priority_key, node_id, version, node in heap
                if self._should_serialize_candidate(
                    depth=depth,
                    node_id=node_id,
                    version=version,
                    node=node,
                )
            ]
            if candidates:
                payloads.append(
                    LinooCandidatesByDepthCheckpointPayload(
                        depth=depth,
                        candidates=sorted(
                            candidates,
                            key=lambda candidate: (
                                -candidate.priority,
                                candidate.node_id,
                                candidate.version,
                            ),
                        ),
                    )
                )
        return payloads

    def _should_serialize_candidate(
        self,
        *,
        depth: int,
        node_id: int,
        version: int,
        node: NodeT,
    ) -> bool:
        """Return whether one heap entry is current enough to checkpoint."""
        node_state = self._node_state_by_id.get(node_id)
        return (
            node_state is not None
            and node_state.status == "frontier"
            and node_state.depth == depth
            and node_state.node is node
            and version == self._candidate_versions_by_node_id.get(node_id, 0)
        )

    def _restore_node_states_from_payload(
        self,
        *,
        tree: trees.Tree[NodeT],
        nodes_by_id: dict[int, NodeT],
        payload: LinooSelectorCheckpointPayload,
    ) -> dict[int, _LinooNodeState[NodeT]]:
        """Restore and validate cached node states from payload."""
        node_states: dict[int, _LinooNodeState[NodeT]] = {}
        seen_node_ids: set[int] = set()
        for node_payload in payload.node_states:
            if node_payload.node_id in seen_node_ids:
                raise _invalid_linoo_checkpoint_payload_error()
            seen_node_ids.add(node_payload.node_id)
            if node_payload.status not in _LINOO_NODE_STATUSES:
                raise _invalid_linoo_checkpoint_payload_error()
            node = nodes_by_id[node_payload.node_id]
            depth = tree.node_depth(node)
            if node_payload.depth != depth:
                raise _invalid_linoo_checkpoint_payload_error()
            status = self._classify_node(node)
            if node_payload.status != status:
                raise _invalid_linoo_checkpoint_payload_error()
            node_states[node.id] = _LinooNodeState(
                node=node,
                depth=depth,
                status=status,
            )
        if set(nodes_by_id) != set(node_states):
            raise _invalid_linoo_checkpoint_payload_error()
        return node_states

    def _depth_stats_from_node_states(
        self,
        node_states: dict[int, _LinooNodeState[NodeT]],
    ) -> dict[int, _LinooDepthStats]:
        """Rebuild depth accounting from restored node states."""
        depth_stats_by_depth: dict[int, _LinooDepthStats] = {}
        for node_state in node_states.values():
            stats = depth_stats_by_depth.setdefault(
                node_state.depth,
                _LinooDepthStats(),
            )
            stats.total_nodes += 1
            stats.increment(node_state.status)
        return depth_stats_by_depth

    def _depth_stats_match_payload(
        self,
        depth_stats_by_depth: dict[int, _LinooDepthStats],
        payloads: list[LinooDepthStatsCheckpointPayload],
    ) -> bool:
        """Return whether restored depth stats exactly match payload stats."""
        payload_by_depth = {payload.depth: payload for payload in payloads}
        if set(payload_by_depth) != set(depth_stats_by_depth):
            return False
        return all(
            stats.total_nodes == payload_by_depth[depth].total_nodes
            and stats.opened_count == payload_by_depth[depth].opened_count
            and stats.frontier_count == payload_by_depth[depth].frontier_count
            and stats.terminal_count == payload_by_depth[depth].terminal_count
            and stats.exact_count == payload_by_depth[depth].exact_count
            and stats.uncached_terminal_candidates
            == payload_by_depth[depth].uncached_terminal_candidates
            and stats.non_openable_count
            == payload_by_depth[depth].non_openable_count
            for depth, stats in depth_stats_by_depth.items()
        )

    def _frontier_ids_from_node_states(
        self,
        node_states: dict[int, _LinooNodeState[NodeT]],
    ) -> dict[int, set[int]]:
        """Rebuild frontier id buckets from restored node states."""
        frontier_ids_by_depth: dict[int, set[int]] = {}
        for node_id, node_state in node_states.items():
            if node_state.status == "frontier":
                frontier_ids_by_depth.setdefault(node_state.depth, set()).add(node_id)
        return frontier_ids_by_depth

    def _restore_candidate_payloads(
        self,
        *,
        tree: trees.Tree[NodeT],
        payload: LinooSelectorCheckpointPayload,
        nodes_by_id: dict[int, NodeT],
    ) -> None:
        """Restore valid candidate heap entries and discard stale ones."""
        for depth_payload in payload.candidates_by_depth:
            for candidate in depth_payload.candidates:
                if candidate.depth != depth_payload.depth:
                    continue
                node = nodes_by_id.get(candidate.node_id)
                if node is None:
                    continue
                node_state = self._node_state_by_id.get(candidate.node_id)
                if (
                    node_state is None
                    or node_state.status != "frontier"
                    or node_state.depth != candidate.depth
                    or tree.node_depth(node) != candidate.depth
                ):
                    continue
                current_version = self._candidate_versions_by_node_id.get(
                    candidate.node_id,
                    candidate.version,
                )
                if candidate.version != current_version:
                    continue
                direct_value = self._direct_value_or_none(node)
                if direct_value is None:
                    continue
                self._candidate_versions_by_node_id[candidate.node_id] = (
                    candidate.version
                )
                self._candidate_signature_by_node_id[candidate.node_id] = (
                    self._candidate_signature(node=node, direct_value=direct_value)
                )
                self._candidate_heap_present_by_node_id[candidate.node_id] = True
                heappush(
                    self._candidates_by_depth.setdefault(candidate.depth, []),
                    (-candidate.priority, candidate.node_id, candidate.version, node),
                )

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
    ) -> int:
        """Refresh cached classifications for one expansion wave."""
        self._cache_tree_identity = id(tree.root_node)
        self._cache_objective_identity = id(objective)
        for node in nodes_to_update.values():
            self._reclassify_node(tree=tree, node=node)
        return len(nodes_to_update)

    def _incremental_nodes_from_expansions(
        self,
        latest_tree_expansions: tree_man.TreeExpansions[NodeT],
    ) -> dict[int, NodeT] | None:
        """Collect touched nodes using the explicit ``TreeExpansions`` API."""
        try:
            creation_expansions = (
                latest_tree_expansions.expansions_with_node_creation
            )
            connection_expansions = (
                latest_tree_expansions.expansions_without_node_creation
            )
            created_nodes = latest_tree_expansions.created_nodes
            affected_child_nodes = latest_tree_expansions.affected_child_nodes
        except AttributeError:
            # Defensive fallback for legacy tests/non-standard selector callers.
            return None
        if not callable(created_nodes) or not callable(affected_child_nodes):
            return None

        nodes_by_id: dict[int, NodeT] = {}
        if self._last_selected_node is not None:
            nodes_by_id[self._last_selected_node.id] = self._last_selected_node

        for expansion in (*creation_expansions, *connection_expansions):
            parent_node = expansion.parent_node
            if parent_node is not None:
                nodes_by_id[parent_node.id] = parent_node
            child_node = expansion.child_node
            nodes_by_id[child_node.id] = child_node

        for node in created_nodes():
            nodes_by_id[node.id] = node
        for node in affected_child_nodes():
            nodes_by_id[node.id] = node
        return nodes_by_id

    def _reclassify_node(self, *, tree: trees.Tree[NodeT], node: NodeT) -> None:
        """Refresh cached state for one node that may have changed."""
        self._remove_node_state(node_id=node.id)
        self._add_node_state(
            node=node,
            depth=tree.node_depth(node),
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
        self._node_state_by_id[node.id] = _LinooNodeState(
            node=node,
            depth=depth,
            status=status,
        )

    def _remove_node_state(self, *, node_id: int) -> None:
        """Remove one cached node classification if it already exists."""
        old_state = self._node_state_by_id.pop(node_id, None)
        if old_state is None:
            return
        depth_stats = self._depth_stats_by_depth[old_state.depth]
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
        if self._direct_value_or_none(node) is None:
            return "uncached_terminal_candidate"
        return "frontier"

    def _mark_candidate_stale(self, node_id: int) -> None:
        """Invalidate existing heap entries for one node lazily."""
        self._candidate_versions_by_node_id[node_id] = (
            self._candidate_versions_by_node_id.get(node_id, 0) + 1
        )
        self._candidate_heap_present_by_node_id[node_id] = False

    def _tree_node_count(self, tree: trees.Tree[NodeT]) -> int | None:
        """Return the tree's known node count, when exposed by the tree object."""
        nodes_count = getattr(tree, "nodes_count", None)
        return nodes_count if isinstance(nodes_count, int) else None

    def _active_frontier_depths(self) -> tuple[int, ...]:
        """Return depths that currently have at least one frontier node."""
        return tuple(
            depth
            for depth, frontier_ids in self._frontier_node_ids_by_depth.items()
            if frontier_ids
        )

    def _choose_depth_from_cache(self) -> int:
        """Choose the active depth from cached counts using Linoo policy."""
        return min(
            self._active_frontier_depths(),
            key=lambda depth: (
                self._depth_stats_by_depth[depth].opened_count * (depth + 1),
                depth,
            ),
        )

    def _frontier_nodes_at_depth(self, depth: int) -> list[NodeT]:
        """Return cached frontier nodes at one depth."""
        return [
            self._node_state_by_id[node_id].node
            for node_id in self._frontier_node_ids_by_depth.get(depth, set())
        ]

    def _register_frontier_candidates(
        self,
        *,
        depth: int,
        nodes: list[NodeT],
        objective: SingleAgentMaxObjective[Any],
    ) -> int:
        """Update cached heap entries for the selected depth only."""
        registered_count = 0
        for node in nodes:
            if self._register_frontier_candidate(
                depth=depth,
                node=node,
                objective=objective,
            ):
                registered_count += 1
        return registered_count

    def _register_frontier_candidate(
        self,
        *,
        depth: int,
        node: NodeT,
        objective: SingleAgentMaxObjective[Any],
    ) -> bool:
        """Push a heap entry when one node is new, changed, or reactivated."""
        direct_value = self._require_direct_value(node)
        node_id = node.id
        signature = self._candidate_signature(node=node, direct_value=direct_value)
        previous_signature = self._candidate_signature_by_node_id.get(node_id)
        heap_entry_present = self._candidate_heap_present_by_node_id.get(
            node_id,
            False,
        )
        if previous_signature == signature and heap_entry_present:
            return False

        version = self._candidate_versions_by_node_id.get(node_id, 0) + 1
        self._candidate_versions_by_node_id[node_id] = version
        self._candidate_signature_by_node_id[node_id] = signature
        self._candidate_heap_present_by_node_id[node_id] = True
        priority = objective.evaluate_value(direct_value, node.state)
        heappush(
            self._candidates_by_depth.setdefault(depth, []),
            (-priority, node_id, version, node),
        )
        return True

    def _choose_best_node_at_depth(
        self,
        *,
        tree: trees.Tree[NodeT],
        depth: int,
    ) -> tuple[NodeT, int]:
        """Return the best cached frontier node, lazily discarding stale entries."""
        heap = self._candidates_by_depth.setdefault(depth, [])
        stale_candidates_skipped = 0
        while heap:
            _priority, node_id, version, node = heap[0]
            current_version = self._candidate_versions_by_node_id.get(node_id)
            if version != current_version:
                heappop(heap)
                stale_candidates_skipped += 1
                continue

            if not self._is_current_frontier_candidate(
                tree=tree,
                node=node,
                expected_depth=depth,
            ):
                heappop(heap)
                self._candidate_heap_present_by_node_id[node_id] = False
                stale_candidates_skipped += 1
                continue

            return node, stale_candidates_skipped

        raise _no_frontier_nodes_error()

    def _is_current_frontier_candidate(
        self,
        *,
        tree: trees.Tree[NodeT],
        node: NodeT,
        expected_depth: int,
    ) -> bool:
        """Return whether one cached node is still selectable frontier."""
        node_state = self._node_state_by_id.get(node.id)
        return (
            node_state is not None
            and node_state.depth == expected_depth
            and node_state.status == "frontier"
            and node_state.node is node
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
        state_rebuilt: bool,
        nodes_incrementally_updated: int,
    ) -> LinooSelectionReport:
        """Build the structured latest-selection table without affecting policy."""
        rows: list[LinooDepthSelectionRow] = []
        selected_depth_selection_index: int | None = None

        for depth, depth_state in depth_stats_by_depth.items():
            opened_count = depth_state.opened_count
            active = depth_state.frontier_count > 0
            selected = depth == selected_depth
            selection_index = opened_count * (depth + 1) if active else None
            row = LinooDepthSelectionRow(
                depth=depth,
                total_nodes=depth_state.total_nodes,
                opened_count=opened_count,
                frontier_count=depth_state.frontier_count,
                terminal_count=depth_state.terminal_count,
                exact_count=depth_state.exact_count,
                uncached_terminal_candidates=(
                    depth_state.uncached_terminal_candidates
                ),
                non_openable_count=depth_state.non_openable_count,
                selection_index=selection_index,
                active=active,
                selected=selected,
            )
            rows.append(row)
            if selected:
                selected_depth_selection_index = selection_index

        sorted_rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.selection_index is None,
                    row.selection_index if row.selection_index is not None else 0,
                    row.depth,
                ),
            )
        )
        if selected_depth_selection_index is None:
            selected_depth_selection_index = (
                depth_stats_by_depth[selected_depth].opened_count
                * (selected_depth + 1)
            )

        return LinooSelectionReport(
            selected_depth=selected_depth,
            selected_node_id=selected_node.id,
            selected_node_direct_value=self._require_direct_value(selected_node).score,
            selected_depth_selection_index=selected_depth_selection_index,
            depth_rows=sorted_rows,
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

    def _candidate_signature(self, *, node: NodeT, direct_value: Value) -> object:
        """Return the cheap invalidation signature for one heap candidate."""
        return (
            id(node),
            direct_value,
            node.all_branches_generated,
            node.tree_evaluation.has_exact_value(),
            self._is_terminal_node(node),
        )
