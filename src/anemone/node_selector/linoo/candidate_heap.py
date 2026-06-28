"""Candidate heap types and diagnostics for Linoo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from anemone.node_evaluation.common.value_candidate import ValueCandidateSource

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
