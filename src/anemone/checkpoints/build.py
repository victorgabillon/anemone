"""Build runtime checkpoint payloads from live Anemone searches."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from random import Random
from time import perf_counter
from typing import TYPE_CHECKING, Any

from .build_atoms import (
    _serialize_branch_collection,
    _serialize_checkpoint_atom_for_build,
    _serialize_evaluation_atom_for_build,
    _serialize_optional_atom,
    _serialize_optional_evaluation_atom,
    _serialize_parent_branches,
)
from .build_context import (
    _CheckpointBuildContext,
    _CheckpointBuildMetrics,
    _log_checkpoint_build_metrics,
    _maybe_log_checkpoint_codec_profile,
    _maybe_reset_checkpoint_profile,
    _NodeCheckpointBuildCache,
    _ParentBranchSerialization,
)
from .build_evaluation_payloads import (
    _build_backup_runtime_payload,
    _build_branch_frontier_payload,
    _build_decision_ordering_payload,
    _build_exploration_index_payload,
    _build_node_evaluation_payload,
    _build_principal_variation_payload,
    _exploration_index_fields,
    _serialize_index_field_value,
)
from .build_node_payloads import (
    _build_node_checkpoint_cache,
    _build_node_payload,
    _build_tree_payload,
    _iter_nodes_in_checkpoint_order,
    _serialize_linked_children,
)
from .build_state_payloads import (
    _build_checkpoint_state_payload,
    _delta_reuse_matches_parent,
    _dump_optional_state_summary,
    _dump_state_parent_branch_for_checkpoint,
    _first_branch_in_stable_order,
    _is_anchor_node,
    _iter_candidate_state_parent_links,
    _preferred_parent_link_from_reusable_state_payload,
    _representative_parent_link,
    _stored_or_first_branch_in_stable_order,
    _try_build_delta_state_payload,
    _try_reuse_checkpoint_state_payload,
)
from .build_values import (
    _serialize_optional_value,
    _serialize_over_event_for_build,
    _serialize_value_for_build,
    _serialize_value_uncached_for_build,
)
from .payloads import SearchRuntimeCheckpointPayload
from .selector_payloads import _build_selector_state_payload
from .tree_expansions_payloads import _build_latest_tree_expansions_payload

if TYPE_CHECKING:
    from anemone.tree_exploration import TreeExploration

    from ._protocols import IncrementalStateCheckpointCodec


def build_search_checkpoint_payload(
    search: TreeExploration[Any],
    *,
    state_codec: IncrementalStateCheckpointCodec[Any],
) -> SearchRuntimeCheckpointPayload:
    """Build a read-only checkpoint payload from one live search runtime.

    The new checkpoint format requires an incremental codec that can emit
    anchor snapshots plus parent-to-child deltas.
    """
    metrics = _CheckpointBuildMetrics()
    context = _CheckpointBuildContext(metrics=metrics)
    _maybe_reset_checkpoint_profile(state_codec)
    tree_payload_started_at = perf_counter()
    tree_payload = _build_tree_payload(
        search=search,
        state_codec=state_codec,
        metrics=metrics,
        context=context,
    )
    metrics.tree_payload_s += perf_counter() - tree_payload_started_at
    rng_state_started_at = perf_counter()
    rng_state = _maybe_dump_rng_state(search)
    metrics.rng_state_total_s += perf_counter() - rng_state_started_at
    latest_tree_expansions_started_at = perf_counter()
    latest_tree_expansions = _build_latest_tree_expansions_payload(search)
    metrics.latest_expansions_total_s += (
        perf_counter() - latest_tree_expansions_started_at
    )
    selector_state_started_at = perf_counter()
    selector_state = _build_selector_state_payload(search)
    metrics.selector_state_total_s += perf_counter() - selector_state_started_at
    _log_checkpoint_build_metrics(metrics)
    _maybe_log_checkpoint_codec_profile(state_codec)
    _maybe_reset_checkpoint_profile(state_codec)
    return SearchRuntimeCheckpointPayload(
        evaluator_version=search.evaluator_version,
        tree=tree_payload,
        rng_state=rng_state,
        latest_tree_expansions=latest_tree_expansions,
        selector_state=selector_state,
    )


def _maybe_dump_rng_state(search: TreeExploration[Any]) -> object | None:
    """Return directly exposed selector RNG state when available."""
    random_generator = getattr(search.node_selector, "random_generator", None)
    if isinstance(random_generator, Random):
        return random_generator.getstate()

    # TreeExploration does not retain the explore() RNG. Selector-specific RNG
    # restoration can be made more complete with selector checkpoint payloads.
    return None


__all__ = [
    "_CheckpointBuildContext",
    "_CheckpointBuildMetrics",
    "_NodeCheckpointBuildCache",
    "_ParentBranchSerialization",
    "_build_backup_runtime_payload",
    "_build_branch_frontier_payload",
    "_build_checkpoint_state_payload",
    "_build_decision_ordering_payload",
    "_build_exploration_index_payload",
    "_build_node_checkpoint_cache",
    "_build_node_evaluation_payload",
    "_build_node_payload",
    "_build_principal_variation_payload",
    "_build_tree_payload",
    "_delta_reuse_matches_parent",
    "_dump_optional_state_summary",
    "_dump_state_parent_branch_for_checkpoint",
    "_exploration_index_fields",
    "_first_branch_in_stable_order",
    "_is_anchor_node",
    "_iter_candidate_state_parent_links",
    "_iter_nodes_in_checkpoint_order",
    "_maybe_dump_rng_state",
    "_preferred_parent_link_from_reusable_state_payload",
    "_representative_parent_link",
    "_serialize_branch_collection",
    "_serialize_checkpoint_atom_for_build",
    "_serialize_evaluation_atom_for_build",
    "_serialize_index_field_value",
    "_serialize_linked_children",
    "_serialize_optional_atom",
    "_serialize_optional_evaluation_atom",
    "_serialize_optional_value",
    "_serialize_over_event_for_build",
    "_serialize_parent_branches",
    "_serialize_value_for_build",
    "_serialize_value_uncached_for_build",
    "_stored_or_first_branch_in_stable_order",
    "_try_build_delta_state_payload",
    "_try_reuse_checkpoint_state_payload",
    "build_search_checkpoint_payload",
]
