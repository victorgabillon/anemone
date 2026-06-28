"""Candidate heap state, operations, and diagnostics for Linoo."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from valanga.evaluations import Value

    from anemone.node_evaluation.common.value_candidate import (
        ValueCandidate,
        ValueCandidateSource,
    )
    from anemone.objectives import SingleAgentMaxObjective

type LinooHeapEntry = tuple[float, int, int]
type LinooRankedCandidate = tuple[float, int, int]


@dataclass(slots=True)
class LinooHeapUpdateDiagnostics:
    """Cheap counters for attributing Linoo heap update cost."""

    candidate_count: int = 0
    push_count: int = 0
    pop_count: int = 0
    stale_skip_count: int = 0
    signature_check_count: int = 0
    signature_recompute_count: int = 0
    version_mismatch_count: int = 0
    priority_state_free_count: int = 0
    priority_stateful_fallback_count: int = 0
    candidate_source_counts: dict[ValueCandidateSource, int] = field(
        default_factory=lambda: cast("dict[ValueCandidateSource, int]", {})
    )
    total_heap_entries: int = 0
    max_heap_size: int = 0
    depth_count: int = 0
    frontier_node_count_seen: int = 0


@dataclass(slots=True)
class LinooCandidateHeap:
    """Mutable candidate heap cache for Linoo frontier nodes."""

    candidates_by_depth: dict[int, list[LinooHeapEntry]] = field(
        default_factory=lambda: cast("dict[int, list[LinooHeapEntry]]", {})
    )
    candidate_versions_by_node_id: dict[int, int] = field(
        default_factory=lambda: cast("dict[int, int]", {})
    )
    candidate_signature_by_node_id: dict[int, object] = field(
        default_factory=lambda: cast("dict[int, object]", {})
    )
    candidate_heap_present_by_node_id: dict[int, bool] = field(
        default_factory=lambda: cast("dict[int, bool]", {})
    )

    def clear(self) -> None:
        """Clear all candidate heap runtime state."""
        self.candidates_by_depth.clear()
        self.candidate_versions_by_node_id.clear()
        self.candidate_signature_by_node_id.clear()
        self.candidate_heap_present_by_node_id.clear()

    def mark_stale(self, node_id: int) -> None:
        """Invalidate existing heap entries for one node lazily."""
        self.candidate_versions_by_node_id[node_id] = (
            self.candidate_versions_by_node_id.get(node_id, 0) + 1
        )
        self.candidate_heap_present_by_node_id[node_id] = False

    def total_heap_entries(self) -> int:
        """Return the total candidate entries currently retained across depths."""
        return sum(len(heap) for heap in self.candidates_by_depth.values())

    def max_heap_size(self) -> int:
        """Return the largest heap size currently retained for one depth."""
        return max((len(heap) for heap in self.candidates_by_depth.values()), default=0)

    def make_update_diagnostics(self) -> LinooHeapUpdateDiagnostics:
        """Return initial heap diagnostics for one candidate update phase."""
        return LinooHeapUpdateDiagnostics(
            depth_count=len(self.candidates_by_depth),
            total_heap_entries=self.total_heap_entries(),
            max_heap_size=self.max_heap_size(),
        )

    def restore_candidate(
        self,
        *,
        depth: int,
        node_id: int,
        priority: float,
        version: int,
        signature: object,
    ) -> None:
        """Restore one validated candidate heap entry from checkpoint payload."""
        self.candidate_versions_by_node_id[node_id] = version
        self.candidate_signature_by_node_id[node_id] = signature
        self.candidate_heap_present_by_node_id[node_id] = True
        heappush(
            self.candidates_by_depth.setdefault(depth, []),
            (-priority, node_id, version),
        )

    def register_frontier_candidates[NodeT](
        self,
        *,
        depth: int,
        nodes: list[NodeT],
        objective: SingleAgentMaxObjective[Any],
        diagnostics: LinooHeapUpdateDiagnostics,
        require_candidate_value: Callable[[NodeT], ValueCandidate],
        candidate_signature: Callable[[NodeT, Value], object],
        candidate_priority: Callable[
            [SingleAgentMaxObjective[Any], Value, NodeT, LinooHeapUpdateDiagnostics],
            float,
        ],
        record_candidate_source: Callable[
            [ValueCandidateSource, LinooHeapUpdateDiagnostics],
            None,
        ],
        diagnostic_phase: Callable[[str], AbstractContextManager[None]],
    ) -> int:
        """Update cached heap entries for the selected depth only."""
        registered_count = 0
        diagnostics.candidate_count += len(nodes)
        for node in nodes:
            if self.register_frontier_candidate(
                depth=depth,
                node=node,
                objective=objective,
                diagnostics=diagnostics,
                require_candidate_value=require_candidate_value,
                candidate_signature=candidate_signature,
                candidate_priority=candidate_priority,
                record_candidate_source=record_candidate_source,
                diagnostic_phase=diagnostic_phase,
            ):
                registered_count += 1
        heap_size = len(self.candidates_by_depth.setdefault(depth, []))
        diagnostics.total_heap_entries = self.total_heap_entries()
        diagnostics.max_heap_size = max(diagnostics.max_heap_size, heap_size)
        return registered_count

    def register_frontier_candidate[NodeT](
        self,
        *,
        depth: int,
        node: NodeT,
        objective: SingleAgentMaxObjective[Any],
        diagnostics: LinooHeapUpdateDiagnostics,
        require_candidate_value: Callable[[NodeT], ValueCandidate],
        candidate_signature: Callable[[NodeT, Value], object],
        candidate_priority: Callable[
            [SingleAgentMaxObjective[Any], Value, NodeT, LinooHeapUpdateDiagnostics],
            float,
        ],
        record_candidate_source: Callable[
            [ValueCandidateSource, LinooHeapUpdateDiagnostics],
            None,
        ],
        diagnostic_phase: Callable[[str], AbstractContextManager[None]],
    ) -> bool:
        """Push a heap entry when one node is new, changed, or reactivated."""
        candidate = require_candidate_value(node)
        candidate_value = cast("Value", candidate.value)
        node_id = cast("int", cast("Any", node).id)
        diagnostics.signature_check_count += 1
        diagnostics.signature_recompute_count += 1
        with diagnostic_phase("select.heap_update.signature"):
            signature = candidate_signature(node, candidate_value)
            previous_signature = self.candidate_signature_by_node_id.get(node_id)
            heap_entry_present = self.candidate_heap_present_by_node_id.get(
                node_id,
                False,
            )
        if previous_signature == signature and heap_entry_present:
            return False

        with diagnostic_phase("select.heap_update.push"):
            version = self.candidate_versions_by_node_id.get(node_id, 0) + 1
            self.candidate_versions_by_node_id[node_id] = version
            self.candidate_signature_by_node_id[node_id] = signature
            self.candidate_heap_present_by_node_id[node_id] = True
            record_candidate_source(candidate.source, diagnostics)
            priority = candidate_priority(objective, candidate_value, node, diagnostics)
            heappush(
                self.candidates_by_depth.setdefault(depth, []),
                (-priority, node_id, version),
            )
            diagnostics.push_count += 1
            diagnostics.max_heap_size = max(
                diagnostics.max_heap_size,
                len(self.candidates_by_depth.setdefault(depth, [])),
            )
        return True

    def rank_valid_candidates_at_depth(
        self,
        *,
        depth: int,
        frontier_node_ids: set[int],
        is_current_frontier_candidate: Callable[[int, int], bool],
        diagnostic_phase: Callable[[str], AbstractContextManager[None]],
        diagnostics: LinooHeapUpdateDiagnostics | None = None,
    ) -> tuple[list[LinooRankedCandidate], int]:
        """Return valid candidates sorted by priority, compacting stale heap entries."""
        heap = self.candidates_by_depth.setdefault(depth, [])
        if diagnostics is not None:
            diagnostics.max_heap_size = max(diagnostics.max_heap_size, len(heap))
        stale_candidates_skipped = 0
        valid_entries_by_node_id: dict[int, LinooHeapEntry] = {}
        while heap:
            with diagnostic_phase("select.heap_update.pop"):
                priority_key, node_id, version = heappop(heap)
            if diagnostics is not None:
                diagnostics.pop_count += 1
            with diagnostic_phase("select.heap_update.stale_check"):
                current_version = self.candidate_versions_by_node_id.get(node_id)
                if version != current_version:
                    stale_candidates_skipped += 1
                    if diagnostics is not None:
                        diagnostics.stale_skip_count += 1
                        diagnostics.version_mismatch_count += 1
                    continue

                if not is_current_frontier_candidate(node_id, depth):
                    self.candidate_heap_present_by_node_id[node_id] = False
                    stale_candidates_skipped += 1
                    if diagnostics is not None:
                        diagnostics.stale_skip_count += 1
                    continue

            valid_entries_by_node_id[node_id] = (priority_key, node_id, version)

        compacted_heap = list(valid_entries_by_node_id.values())
        for node_id in tuple(frontier_node_ids):
            self.candidate_heap_present_by_node_id[node_id] = (
                node_id in valid_entries_by_node_id
            )
        self.candidates_by_depth[depth] = compacted_heap
        heapify(self.candidates_by_depth[depth])
        if diagnostics is not None:
            diagnostics.total_heap_entries = self.total_heap_entries()
            diagnostics.max_heap_size = max(
                diagnostics.max_heap_size,
                len(self.candidates_by_depth[depth]),
            )

        ranked_candidates = sorted(
            compacted_heap,
            key=lambda entry: (entry[0], entry[1]),
        )
        return ranked_candidates, stale_candidates_skipped
