"""Decode sharded checkpoint shards into typed checkpoint payloads."""

# pylint: disable=duplicate-code

from __future__ import annotations

# pyright: reportPrivateUsage=false
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._json_types import (
    optional_int_field,
    optional_list_field,
    optional_mapping_field,
    optional_str_field,
    require_bool,
    require_bool_field,
    require_float_field,
    require_int,
    require_int_field,
    require_mapping,
    require_mapping_field,
    require_str_field,
)
from .payloads import (
    AlgorithmNodeCheckpointPayload,
    AnchorCheckpointStatePayload,
    BackupRuntimeCheckpointPayload,
    BranchFrontierCheckpointPayload,
    BranchOrderingCheckpointPayload,
    CheckpointNodeStatePayload,
    DecisionOrderingCheckpointPayload,
    DeltaCheckpointStatePayload,
    ExplorationIndexCheckpointPayload,
    LinkedChildCheckpointPayload,
    LinooCandidateCheckpointPayload,
    LinooCandidatesByDepthCheckpointPayload,
    LinooDepthStatsCheckpointPayload,
    LinooNodeStateCheckpointPayload,
    LinooSelectorCheckpointPayload,
    NodeEvaluationCheckpointPayload,
    PrincipalVariationCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    SerializedOverEventPayload,
    SerializedValuePayload,
    TreeCheckpointPayload,
    TreeExpansionCheckpointPayload,
    TreeExpansionsCheckpointPayload,
)
from .sharded_errors import ShardedCheckpointManifestError
from .sharded_io import (
    _group_manifest_shards,
    _load_mapping_json_shard,
    _only_manifest_shard,
    _optional_mapping_json_shard,
    _read_jsonl_shard_records,
    _required_manifest_shards,
)
from .sharded_manifest import read_sharded_checkpoint_manifest

if TYPE_CHECKING:
    from typing import Literal

    from ._protocols import CheckpointStateSummary
    from .payloads import CheckpointAtomPayload


def load_sharded_search_checkpoint(
    checkpoint_dir: str | Path,
) -> SearchRuntimeCheckpointPayload:
    """Load a C2b sharded runtime checkpoint as a typed checkpoint payload.

    C2c intentionally reconstructs the monolithic typed payload first. Later
    streaming restore work can replace this internal assembly path without
    changing the C2b manifest contract.
    """
    resolved_checkpoint_dir = Path(checkpoint_dir)
    manifest = read_sharded_checkpoint_manifest(
        resolved_checkpoint_dir / "manifest.json"
    )
    grouped_shards = _group_manifest_shards(manifest)

    metadata = _load_mapping_json_shard(
        resolved_checkpoint_dir,
        _only_manifest_shard(grouped_shards, "metadata"),
    )
    node_records = [
        record
        for shard in _required_manifest_shards(grouped_shards, "node_records")
        for record in _read_jsonl_shard_records(resolved_checkpoint_dir, shard)
    ]
    if len(node_records) != manifest.total_node_count:
        raise ShardedCheckpointManifestError.node_count_mismatch()
    if manifest.total_branch_count is not None:
        branch_count = sum(
            len(optional_list_field(record, "linked_children"))
            for record in node_records
        )
        if branch_count != manifest.total_branch_count:
            raise ShardedCheckpointManifestError.branch_count_mismatch()

    selector_state = _optional_mapping_json_shard(
        resolved_checkpoint_dir,
        grouped_shards,
        "selector",
    )
    latest_tree_expansions = _optional_mapping_json_shard(
        resolved_checkpoint_dir,
        grouped_shards,
        "latest_expansions",
    )

    return SearchRuntimeCheckpointPayload(
        format_version=require_int_field(metadata, "format_version"),
        evaluator_version=require_int_field(metadata, "evaluator_version"),
        rng_state=metadata.get("rng_state"),
        tree=TreeCheckpointPayload(
            root_node_id=require_int_field(metadata, "root_node_id"),
            nodes=[
                _algorithm_node_payload_from_mapping(record) for record in node_records
            ],
        ),
        latest_tree_expansions=(
            None
            if latest_tree_expansions is None
            else _tree_expansions_payload_from_mapping(latest_tree_expansions)
        ),
        selector_state=(
            None
            if selector_state is None
            else _selector_payload_from_mapping(selector_state)
        ),
    )


def _algorithm_node_payload_from_mapping(
    payload: dict[str, object],
) -> AlgorithmNodeCheckpointPayload:
    """Build one typed node payload from a C2b JSON record."""
    return AlgorithmNodeCheckpointPayload(
        node_id=require_int_field(payload, "node_id"),
        parent_node_id=optional_int_field(payload, "parent_node_id"),
        branch_from_parent=cast(
            "CheckpointAtomPayload | None",
            payload.get("branch_from_parent"),
        ),
        depth=require_int_field(payload, "depth"),
        state_payload=_state_payload_from_mapping(
            require_mapping_field(payload, "state_payload")
        ),
        generated_all_branches=require_bool_field(payload, "generated_all_branches"),
        unopened_branches=cast(
            "list[CheckpointAtomPayload]",
            optional_list_field(payload, "unopened_branches"),
        ),
        linked_children=[
            _linked_child_payload_from_mapping(
                require_mapping(child, field_name="linked_children[]")
            )
            for child in optional_list_field(payload, "linked_children")
        ],
        evaluation=(
            None
            if (evaluation := optional_mapping_field(payload, "evaluation")) is None
            else _node_evaluation_payload_from_mapping(evaluation)
        ),
        exploration_index=(
            None
            if (
                exploration_index := optional_mapping_field(
                    payload,
                    "exploration_index",
                )
            )
            is None
            else _exploration_index_payload_from_mapping(exploration_index)
        ),
    )


def _state_payload_from_mapping(
    payload: dict[str, object],
) -> CheckpointNodeStatePayload:
    """Build one typed checkpoint state payload from a JSON object."""
    if "anchor_ref" in payload:
        return AnchorCheckpointStatePayload(
            anchor_ref=payload["anchor_ref"],
            state_summary=cast(
                "CheckpointStateSummary | None", payload.get("state_summary")
            ),
        )
    return DeltaCheckpointStatePayload(
        state_parent_node_id=require_int_field(payload, "state_parent_node_id"),
        state_parent_branch=cast(
            "CheckpointAtomPayload | None",
            payload.get("state_parent_branch"),
        ),
        delta_ref=payload["delta_ref"],
        state_summary=cast(
            "CheckpointStateSummary | None", payload.get("state_summary")
        ),
    )


def _linked_child_payload_from_mapping(
    payload: dict[str, object],
) -> LinkedChildCheckpointPayload:
    """Build one typed linked-child payload from a JSON object."""
    return LinkedChildCheckpointPayload(
        branch_key=cast("CheckpointAtomPayload", payload["branch_key"]),
        child_node_id=require_int_field(payload, "child_node_id"),
    )


def _node_evaluation_payload_from_mapping(
    payload: dict[str, object],
) -> NodeEvaluationCheckpointPayload:
    """Build one typed node-evaluation payload from a JSON object."""
    return NodeEvaluationCheckpointPayload(
        direct_value=_optional_serialized_value(payload.get("direct_value")),
        direct_evaluation_version=optional_int_field(
            payload,
            "direct_evaluation_version",
        ),
        backed_up_value=_optional_serialized_value(payload.get("backed_up_value")),
        decision_ordering=(
            None
            if (
                decision_ordering := optional_mapping_field(
                    payload,
                    "decision_ordering",
                )
            )
            is None
            else _decision_ordering_payload_from_mapping(decision_ordering)
        ),
        principal_variation=(
            None
            if (
                principal_variation := optional_mapping_field(
                    payload,
                    "principal_variation",
                )
            )
            is None
            else _principal_variation_payload_from_mapping(principal_variation)
        ),
        branch_frontier=(
            None
            if (branch_frontier := optional_mapping_field(payload, "branch_frontier"))
            is None
            else BranchFrontierCheckpointPayload(
                frontier_branches=cast(
                    "list[CheckpointAtomPayload]",
                    optional_list_field(branch_frontier, "frontier_branches"),
                )
            )
        ),
        backup_runtime=(
            None
            if (backup_runtime := optional_mapping_field(payload, "backup_runtime"))
            is None
            else _backup_runtime_payload_from_mapping(backup_runtime)
        ),
    )


def _optional_serialized_value(payload: object) -> SerializedValuePayload | None:
    """Build an optional typed serialized value from a JSON object."""
    if payload is None:
        return None
    return _serialized_value_payload_from_mapping(
        require_mapping(payload, field_name="serialized_value")
    )


def _serialized_value_payload_from_mapping(
    payload: dict[str, object],
) -> SerializedValuePayload:
    """Build one typed serialized value payload from a JSON object."""
    over_event = optional_mapping_field(payload, "over_event")
    return SerializedValuePayload(
        score=require_float_field(payload, "score"),
        certainty=require_str_field(payload, "certainty"),
        over_event=(
            None
            if over_event is None
            else SerializedOverEventPayload(
                outcome=require_str_field(over_event, "outcome"),
                termination=cast(
                    "CheckpointAtomPayload",
                    over_event.get("termination"),
                ),
                winner=cast(
                    "CheckpointAtomPayload",
                    over_event.get("winner"),
                ),
            )
        ),
        line=cast("list[CheckpointAtomPayload] | None", payload.get("line")),
    )


def _decision_ordering_payload_from_mapping(
    payload: dict[str, object],
) -> DecisionOrderingCheckpointPayload:
    """Build one typed decision-ordering payload from a JSON object."""
    return DecisionOrderingCheckpointPayload(
        branch_ordering=[
            BranchOrderingCheckpointPayload(
                branch_key=cast("CheckpointAtomPayload", ordering["branch_key"]),
                primary_score=require_float_field(ordering, "primary_score"),
                tactical_tiebreak=require_int_field(ordering, "tactical_tiebreak"),
                stable_tiebreak_id=require_int_field(ordering, "stable_tiebreak_id"),
            )
            for ordering in _mapping_items(payload, "branch_ordering")
        ]
    )


def _principal_variation_payload_from_mapping(
    payload: dict[str, object],
) -> PrincipalVariationCheckpointPayload:
    """Build one typed principal-variation payload from a JSON object."""
    return PrincipalVariationCheckpointPayload(
        best_branch_sequence=cast(
            "list[CheckpointAtomPayload]",
            optional_list_field(payload, "best_branch_sequence"),
        ),
        pv_version=require_int(payload.get("pv_version", 0), field_name="pv_version"),
        cached_best_child_version=optional_int_field(
            payload,
            "cached_best_child_version",
        ),
    )


def _backup_runtime_payload_from_mapping(
    payload: dict[str, object],
) -> BackupRuntimeCheckpointPayload:
    """Build one typed backup-runtime payload from a JSON object."""
    return BackupRuntimeCheckpointPayload(
        best_branch=cast("CheckpointAtomPayload | None", payload.get("best_branch")),
        second_best_branch=cast(
            "CheckpointAtomPayload | None",
            payload.get("second_best_branch"),
        ),
        exact_child_count=require_int(
            payload.get("exact_child_count", 0),
            field_name="exact_child_count",
        ),
        selected_child_pv_version=optional_int_field(
            payload,
            "selected_child_pv_version",
        ),
        is_initialized=require_bool(
            payload.get("is_initialized", False),
            field_name="is_initialized",
        ),
    )


def _exploration_index_payload_from_mapping(
    payload: dict[str, object],
) -> ExplorationIndexCheckpointPayload:
    """Build one typed exploration-index payload from a JSON object."""
    return ExplorationIndexCheckpointPayload(
        kind=optional_str_field(payload, "kind"),
        payload=payload.get("payload"),
    )


def _tree_expansions_payload_from_mapping(
    payload: dict[str, object],
) -> TreeExpansionsCheckpointPayload:
    """Build typed latest-expansions payload from a JSON object."""
    return TreeExpansionsCheckpointPayload(
        expansions_with_node_creation=[
            _tree_expansion_payload_from_mapping(expansion)
            for expansion in _mapping_items(payload, "expansions_with_node_creation")
        ],
        expansions_without_node_creation=[
            _tree_expansion_payload_from_mapping(expansion)
            for expansion in _mapping_items(payload, "expansions_without_node_creation")
        ],
    )


def _tree_expansion_payload_from_mapping(
    payload: dict[str, object],
) -> TreeExpansionCheckpointPayload:
    """Build one typed tree-expansion payload from a JSON object."""
    return TreeExpansionCheckpointPayload(
        child_node_id=require_int_field(payload, "child_node_id"),
        parent_node_id=optional_int_field(payload, "parent_node_id"),
        branch_key=cast("CheckpointAtomPayload | None", payload.get("branch_key")),
        creation_child_node=require_bool_field(payload, "creation_child_node"),
    )


def _selector_payload_from_mapping(
    payload: dict[str, object],
) -> LinooSelectorCheckpointPayload:
    """Build the generic selector payload currently supported by Anemone."""
    return LinooSelectorCheckpointPayload(
        type=cast('Literal["linoo"]', payload.get("type", "linoo")),
        version=require_int(payload.get("version", 1), field_name="version"),
        depth_stats=[
            _linoo_depth_stats_payload_from_mapping(depth_stats)
            for depth_stats in _mapping_items(payload, "depth_stats")
        ],
        node_states=[
            LinooNodeStateCheckpointPayload(
                node_id=require_int_field(node_state, "node_id"),
                depth=require_int_field(node_state, "depth"),
                status=require_str_field(node_state, "status"),
            )
            for node_state in _mapping_items(payload, "node_states")
        ],
        candidates_by_depth=[
            _linoo_candidates_by_depth_payload_from_mapping(candidates_by_depth)
            for candidates_by_depth in _mapping_items(payload, "candidates_by_depth")
        ],
        last_selected_node_id=optional_int_field(payload, "last_selected_node_id"),
    )


def _linoo_depth_stats_payload_from_mapping(
    payload: dict[str, object],
) -> LinooDepthStatsCheckpointPayload:
    """Build one typed Linoo depth-stats payload from a JSON object."""
    return LinooDepthStatsCheckpointPayload(
        depth=require_int_field(payload, "depth"),
        total_nodes=require_int_field(payload, "total_nodes"),
        opened_count=require_int_field(payload, "opened_count"),
        frontier_count=require_int_field(payload, "frontier_count"),
        terminal_count=require_int_field(payload, "terminal_count"),
        exact_count=require_int_field(payload, "exact_count"),
        uncached_terminal_candidates=require_int_field(
            payload,
            "uncached_terminal_candidates",
        ),
        non_openable_count=require_int_field(payload, "non_openable_count"),
    )


def _linoo_candidates_by_depth_payload_from_mapping(
    payload: dict[str, object],
) -> LinooCandidatesByDepthCheckpointPayload:
    """Build one typed Linoo candidates-by-depth payload from a JSON object."""
    return LinooCandidatesByDepthCheckpointPayload(
        depth=require_int_field(payload, "depth"),
        candidates=[
            LinooCandidateCheckpointPayload(
                node_id=require_int_field(candidate, "node_id"),
                depth=require_int_field(candidate, "depth"),
                priority=require_float_field(candidate, "priority"),
                version=require_int_field(candidate, "version"),
            )
            for candidate in _mapping_items(payload, "candidates")
        ],
    )


def _mapping_items(
    payload: dict[str, object],
    key: str,
) -> list[dict[str, object]]:
    """Return optional list entries as mappings."""
    return [
        require_mapping(item, field_name=f"{key}[]")
        for item in optional_list_field(payload, key)
    ]
