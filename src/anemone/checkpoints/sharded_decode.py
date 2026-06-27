"""Decode sharded checkpoint shards into typed checkpoint payloads."""

# pylint: disable=duplicate-code

from __future__ import annotations

# pyright: reportPrivateUsage=false
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
            len(cast("list[object]", record.get("linked_children", [])))
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
        format_version=cast("int", metadata["format_version"]),
        evaluator_version=cast("int", metadata["evaluator_version"]),
        rng_state=metadata.get("rng_state"),
        tree=TreeCheckpointPayload(
            root_node_id=cast("int", metadata["root_node_id"]),
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
        node_id=cast("int", payload["node_id"]),
        parent_node_id=cast("int | None", payload["parent_node_id"]),
        branch_from_parent=cast(
            "CheckpointAtomPayload | None",
            payload["branch_from_parent"],
        ),
        depth=cast("int", payload["depth"]),
        state_payload=_state_payload_from_mapping(
            cast("dict[str, object]", payload["state_payload"])
        ),
        generated_all_branches=cast("bool", payload["generated_all_branches"]),
        unopened_branches=cast(
            "list[CheckpointAtomPayload]",
            payload.get("unopened_branches", []),
        ),
        linked_children=[
            _linked_child_payload_from_mapping(cast("dict[str, object]", child))
            for child in cast("list[object]", payload.get("linked_children", []))
        ],
        evaluation=(
            None
            if payload.get("evaluation") is None
            else _node_evaluation_payload_from_mapping(
                cast("dict[str, object]", payload["evaluation"])
            )
        ),
        exploration_index=(
            None
            if payload.get("exploration_index") is None
            else _exploration_index_payload_from_mapping(
                cast("dict[str, object]", payload["exploration_index"])
            )
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
        state_parent_node_id=cast("int", payload["state_parent_node_id"]),
        state_parent_branch=cast(
            "CheckpointAtomPayload | None",
            payload["state_parent_branch"],
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
        child_node_id=cast("int", payload["child_node_id"]),
    )


def _node_evaluation_payload_from_mapping(
    payload: dict[str, object],
) -> NodeEvaluationCheckpointPayload:
    """Build one typed node-evaluation payload from a JSON object."""
    return NodeEvaluationCheckpointPayload(
        direct_value=_optional_serialized_value(payload.get("direct_value")),
        direct_evaluation_version=cast(
            "int | None",
            payload.get("direct_evaluation_version"),
        ),
        backed_up_value=_optional_serialized_value(payload.get("backed_up_value")),
        decision_ordering=(
            None
            if payload.get("decision_ordering") is None
            else _decision_ordering_payload_from_mapping(
                cast("dict[str, object]", payload["decision_ordering"])
            )
        ),
        principal_variation=(
            None
            if payload.get("principal_variation") is None
            else _principal_variation_payload_from_mapping(
                cast("dict[str, object]", payload["principal_variation"])
            )
        ),
        branch_frontier=(
            None
            if payload.get("branch_frontier") is None
            else BranchFrontierCheckpointPayload(
                frontier_branches=cast(
                    "list[CheckpointAtomPayload]",
                    cast("dict[str, object]", payload["branch_frontier"]).get(
                        "frontier_branches",
                        [],
                    ),
                )
            )
        ),
        backup_runtime=(
            None
            if payload.get("backup_runtime") is None
            else _backup_runtime_payload_from_mapping(
                cast("dict[str, object]", payload["backup_runtime"])
            )
        ),
    )


def _optional_serialized_value(payload: object) -> SerializedValuePayload | None:
    """Build an optional typed serialized value from a JSON object."""
    if payload is None:
        return None
    return _serialized_value_payload_from_mapping(cast("dict[str, object]", payload))


def _serialized_value_payload_from_mapping(
    payload: dict[str, object],
) -> SerializedValuePayload:
    """Build one typed serialized value payload from a JSON object."""
    return SerializedValuePayload(
        score=cast("float", payload["score"]),
        certainty=cast("str", payload["certainty"]),
        over_event=(
            None
            if payload.get("over_event") is None
            else SerializedOverEventPayload(
                outcome=cast(
                    "str",
                    cast("dict[str, object]", payload["over_event"])["outcome"],
                ),
                termination=cast(
                    "CheckpointAtomPayload",
                    cast("dict[str, object]", payload["over_event"]).get("termination"),
                ),
                winner=cast(
                    "CheckpointAtomPayload",
                    cast("dict[str, object]", payload["over_event"]).get("winner"),
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
                primary_score=cast("float", ordering["primary_score"]),
                tactical_tiebreak=cast("int", ordering["tactical_tiebreak"]),
                stable_tiebreak_id=cast("int", ordering["stable_tiebreak_id"]),
            )
            for ordering in cast(
                "list[dict[str, object]]",
                payload.get("branch_ordering", []),
            )
        ]
    )


def _principal_variation_payload_from_mapping(
    payload: dict[str, object],
) -> PrincipalVariationCheckpointPayload:
    """Build one typed principal-variation payload from a JSON object."""
    return PrincipalVariationCheckpointPayload(
        best_branch_sequence=cast(
            "list[CheckpointAtomPayload]",
            payload.get("best_branch_sequence", []),
        ),
        pv_version=cast("int", payload.get("pv_version", 0)),
        cached_best_child_version=cast(
            "int | None",
            payload.get("cached_best_child_version"),
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
        exact_child_count=cast("int", payload.get("exact_child_count", 0)),
        selected_child_pv_version=cast(
            "int | None",
            payload.get("selected_child_pv_version"),
        ),
        is_initialized=cast("bool", payload.get("is_initialized", False)),
    )


def _exploration_index_payload_from_mapping(
    payload: dict[str, object],
) -> ExplorationIndexCheckpointPayload:
    """Build one typed exploration-index payload from a JSON object."""
    return ExplorationIndexCheckpointPayload(
        kind=cast("str | None", payload.get("kind")),
        payload=payload.get("payload"),
    )


def _tree_expansions_payload_from_mapping(
    payload: dict[str, object],
) -> TreeExpansionsCheckpointPayload:
    """Build typed latest-expansions payload from a JSON object."""
    return TreeExpansionsCheckpointPayload(
        expansions_with_node_creation=[
            _tree_expansion_payload_from_mapping(cast("dict[str, object]", expansion))
            for expansion in cast(
                "list[object]",
                payload.get("expansions_with_node_creation", []),
            )
        ],
        expansions_without_node_creation=[
            _tree_expansion_payload_from_mapping(cast("dict[str, object]", expansion))
            for expansion in cast(
                "list[object]",
                payload.get("expansions_without_node_creation", []),
            )
        ],
    )


def _tree_expansion_payload_from_mapping(
    payload: dict[str, object],
) -> TreeExpansionCheckpointPayload:
    """Build one typed tree-expansion payload from a JSON object."""
    return TreeExpansionCheckpointPayload(
        child_node_id=cast("int", payload["child_node_id"]),
        parent_node_id=cast("int | None", payload["parent_node_id"]),
        branch_key=cast("CheckpointAtomPayload | None", payload["branch_key"]),
        creation_child_node=cast("bool", payload["creation_child_node"]),
    )


def _selector_payload_from_mapping(
    payload: dict[str, object],
) -> LinooSelectorCheckpointPayload:
    """Build the generic selector payload currently supported by Anemone."""
    return LinooSelectorCheckpointPayload(
        type=cast('Literal["linoo"]', payload.get("type", "linoo")),
        version=cast("int", payload.get("version", 1)),
        depth_stats=[
            LinooDepthStatsCheckpointPayload(
                depth=cast("int", depth_stats["depth"]),
                total_nodes=cast("int", depth_stats["total_nodes"]),
                opened_count=cast("int", depth_stats["opened_count"]),
                frontier_count=cast("int", depth_stats["frontier_count"]),
                terminal_count=cast("int", depth_stats["terminal_count"]),
                exact_count=cast("int", depth_stats["exact_count"]),
                uncached_terminal_candidates=cast(
                    "int",
                    depth_stats["uncached_terminal_candidates"],
                ),
                non_openable_count=cast("int", depth_stats["non_openable_count"]),
            )
            for depth_stats in cast(
                "list[dict[str, object]]",
                payload.get("depth_stats", []),
            )
        ],
        node_states=[
            LinooNodeStateCheckpointPayload(
                node_id=cast("int", node_state["node_id"]),
                depth=cast("int", node_state["depth"]),
                status=cast("str", node_state["status"]),
            )
            for node_state in cast(
                "list[dict[str, object]]",
                payload.get("node_states", []),
            )
        ],
        candidates_by_depth=[
            _linoo_candidates_by_depth_payload_from_mapping(candidates_by_depth)
            for candidates_by_depth in cast(
                "list[dict[str, object]]",
                payload.get("candidates_by_depth", []),
            )
        ],
        last_selected_node_id=cast("int | None", payload.get("last_selected_node_id")),
    )


def _linoo_candidates_by_depth_payload_from_mapping(
    payload: dict[str, object],
) -> LinooCandidatesByDepthCheckpointPayload:
    """Build one typed Linoo candidates-by-depth payload from a JSON object."""
    return LinooCandidatesByDepthCheckpointPayload(
        depth=cast("int", payload["depth"]),
        candidates=[
            LinooCandidateCheckpointPayload(
                node_id=cast("int", candidate["node_id"]),
                depth=cast("int", candidate["depth"]),
                priority=cast("float", candidate["priority"]),
                version=cast("int", candidate["version"]),
            )
            for candidate in cast(
                "list[dict[str, object]]",
                payload.get("candidates", []),
            )
        ],
    )
