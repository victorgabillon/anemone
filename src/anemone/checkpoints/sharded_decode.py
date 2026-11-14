"""Decode sharded checkpoint shards into typed checkpoint payloads."""

# pylint: disable=duplicate-code

# pyright: reportPrivateUsage=false
import json
from dataclasses import dataclass, fields, is_dataclass
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
    require_list,
    require_mapping,
    require_mapping_field,
    require_str_field,
)
from .metadata import _validate_payload
from .payloads import (
    AlgorithmNodeCheckpointPayload,
    AnchorCheckpointStatePayload,
    BackupRuntimeCheckpointPayload,
    BranchFrontierCheckpointPayload,
    BranchOrderingCheckpointPayload,
    CheckpointAtomPayload,
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
from .sharded_io import _group_manifest_shards
from .sharded_manifest import sharded_checkpoint_manifest_from_jsonable
from .sharded_read import _ShardReader
from .sharded_read_limits import ShardedCheckpointReadLimits  # noqa: TC001
from .sharded_types import (  # noqa: TC001
    ShardedCheckpointManifest,
    ShardedCheckpointShardRef,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal

    from ._protocols import CheckpointStateSummary


def load_sharded_search_checkpoint(
    checkpoint_dir: str | Path,
    *,
    limits: ShardedCheckpointReadLimits | None = None,
) -> SearchRuntimeCheckpointPayload:
    """Materialize exactly one native layout without decoding states or a runtime.

    Explicit limits are required at prepared-consumer boundaries. ``None`` keeps
    the ordinary unbounded contract. Complete typed storage is O(N + E + P).
    """
    reader = _ShardReader(Path(checkpoint_dir), limits)
    manifest = sharded_checkpoint_manifest_from_jsonable(
        reader.mapping("manifest.json")
    )
    reader.bound(len(manifest.shards), "max_shards")
    reader.bound(manifest.total_node_count, "max_nodes")
    reader.bound(manifest.total_branch_count or 0, "max_branches")
    grouped = _group_manifest_shards(manifest)
    if grouped.get("node_records") and any(
        grouped.get(kind) for kind in ("node_shells", "state_payloads", "node_runtime")
    ):
        raise ShardedCheckpointManifestError.invalid_input("mixed_layouts")
    paths = [shard.path for shard in manifest.shards]
    if len(set(paths)) != len(paths) or "manifest.json" in paths:
        raise ShardedCheckpointManifestError.invalid_input("duplicate_shard_path")
    metadata_shard = grouped["metadata"][0]
    metadata = reader.mapping(metadata_shard.path, metadata_shard)
    split = not grouped.get("node_records")
    if split and set(metadata) != {
        "format_version",
        "evaluator_version",
        "rng_state",
        "rollout_rng_state",
        "root_node_id",
        "node_count",
        "branch_count",
    }:
        raise ShardedCheckpointManifestError.invalid_input("incomplete_split_metadata")
    if grouped.get("node_records"):
        nodes: list[AlgorithmNodeCheckpointPayload] = []
        branches = 0
        for shard in grouped["node_records"]:
            for record in reader.records(shard):
                reader.bound(len(nodes) + 1, "max_nodes")
                if len(nodes) >= manifest.total_node_count:
                    raise ShardedCheckpointManifestError.node_count_mismatch()
                branches += len(optional_list_field(record, "linked_children"))
                reader.bound(branches, "max_branches")
                nodes.append(_algorithm_node_payload_from_mapping(record))
    else:
        nodes = _split_node_payloads(
            reader,
            manifest,
            grouped["state_payloads"],
            grouped["node_runtime"],
            grouped["node_shells"],
        )
    if len(nodes) != manifest.total_node_count:
        raise ShardedCheckpointManifestError.node_count_mismatch()
    branches = sum(len(node.linked_children) for node in nodes)
    if (
        manifest.total_branch_count is not None
        and branches != manifest.total_branch_count
    ):
        raise ShardedCheckpointManifestError.branch_count_mismatch()
    reader.bound(branches, "max_branches")
    for name, actual in (("node_count", len(nodes)), ("branch_count", branches)):
        if name in metadata and require_int_field(metadata, name) != actual:
            raise ShardedCheckpointManifestError.invalid_input(
                "metadata_count_mismatch"
            )
    selector = None
    expansions = None
    if grouped.get("selector"):
        shard = grouped["selector"][0]
        raw = reader.mapping(shard.path, shard)
        selector = _selector_payload_from_mapping(raw)
        if split:
            _require_stored_fields(raw, selector)
    if grouped.get("latest_expansions"):
        shard = grouped["latest_expansions"][0]
        raw = reader.mapping(shard.path, shard)
        expansions = _tree_expansions_payload_from_mapping(raw)
        if split:
            _require_stored_fields(raw, expansions)
    payload = SearchRuntimeCheckpointPayload(
        format_version=require_int_field(metadata, "format_version"),
        evaluator_version=require_int_field(metadata, "evaluator_version"),
        rng_state=metadata.get("rng_state"),
        rollout_rng_state=metadata.get("rollout_rng_state"),
        tree=TreeCheckpointPayload(
            root_node_id=require_int_field(metadata, "root_node_id"), nodes=nodes
        ),
        latest_tree_expansions=expansions,
        selector_state=selector,
    )
    _validate_payload(payload)
    ids = {node.node_id for node in nodes}
    for node in nodes:
        reader.check()
        if any(link.child_node_id not in ids for link in node.linked_children):
            raise ShardedCheckpointManifestError.invalid_input("unknown_linked_child")
    reader.check()
    return payload


@dataclass(slots=True)
class _NodeRuntime:
    unopened_branches: list[CheckpointAtomPayload]
    linked_children: list[LinkedChildCheckpointPayload]
    evaluation: NodeEvaluationCheckpointPayload | None
    exploration_index: ExplorationIndexCheckpointPayload | None


def _require_stored_fields(raw: object, typed: object) -> None:
    """Refuse schema omissions rather than silently installing decoder defaults."""
    if is_dataclass(typed) and not isinstance(typed, type):
        mapping = require_mapping(raw, field_name="split_record")
        names = {field.name for field in fields(typed)}
        if set(mapping) != names:
            raise ShardedCheckpointManifestError.invalid_input(
                "incomplete_or_unknown_typed_fields"
            )
        for name in names:
            _require_stored_fields(mapping[name], getattr(typed, name))
    elif isinstance(typed, list):
        items = require_list(raw, field_name="split_list")
        typed_items = cast("list[object]", typed)
        if len(items) != len(typed_items):
            raise ShardedCheckpointManifestError.invalid_input("split_list_coverage")
        for item, value in zip(items, typed_items, strict=True):
            _require_stored_fields(item, value)


def _runtime_from_mapping(payload: dict[str, object]) -> _NodeRuntime:
    evaluation = optional_mapping_field(payload, "evaluation")
    index = optional_mapping_field(payload, "exploration_index")
    return _NodeRuntime(
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
        evaluation=None
        if evaluation is None
        else _node_evaluation_payload_from_mapping(evaluation),
        exploration_index=None
        if index is None
        else _exploration_index_payload_from_mapping(index),
    )


def _split_records(
    reader: _ShardReader,
    shards: list[ShardedCheckpointShardRef],
    total: int,
    expected_fields: set[str],
) -> Iterator[tuple[int, dict[str, object]]]:
    """Check exact family shape, IDs and counts before retaining typed records."""
    seen: set[int] = set()
    for shard in shards:
        for record in reader.records(shard):
            node_id = require_int_field(record, "node_id")
            if node_id in seen or set(record) != expected_fields:
                raise ShardedCheckpointManifestError.invalid_input(
                    "duplicate_ID_or_incomplete_split_record"
                )
            reader.bound(len(seen) + 1, "max_nodes")
            if len(seen) >= total:
                raise ShardedCheckpointManifestError.node_count_mismatch()
            seen.add(node_id)
            yield node_id, record
    if len(seen) != total:
        raise ShardedCheckpointManifestError.node_count_mismatch()


def _split_node_payloads(
    reader: _ShardReader,
    manifest: ShardedCheckpointManifest,
    state_shards: list[ShardedCheckpointShardRef],
    runtime_shards: list[ShardedCheckpointShardRef],
    shell_shards: list[ShardedCheckpointShardRef],
) -> list[AlgorithmNodeCheckpointPayload]:
    """Join typed state/runtime records, consuming each as shells arrive in order."""
    states: dict[int, CheckpointNodeStatePayload] = {}
    runtimes: dict[int, _NodeRuntime] = {}
    total = manifest.total_node_count
    for node_id, record in _split_records(
        reader, state_shards, total, {"node_id", "state_payload"}
    ):
        state = require_mapping_field(record, "state_payload")
        expected = (
            {"anchor_ref", "state_summary"}
            if "anchor_ref" in state
            else {
                "state_parent_node_id",
                "state_parent_branch",
                "delta_ref",
                "state_summary",
            }
        )
        if set(state) != expected:
            raise ShardedCheckpointManifestError.invalid_input(
                "ambiguous_or_incomplete_state_payload"
            )
        states[node_id] = _state_payload_from_mapping(state)
        _require_stored_fields(state, states[node_id])
    branches = 0
    for node_id, record in _split_records(
        reader,
        runtime_shards,
        total,
        {
            "node_id",
            "unopened_branches",
            "linked_children",
            "evaluation",
            "exploration_index",
        },
    ):
        if node_id not in states:
            raise ShardedCheckpointManifestError.invalid_input(
                "split_node_ID_coverage_mismatch"
            )
        branches += len(optional_list_field(record, "linked_children"))
        reader.bound(branches, "max_branches")
        runtimes[node_id] = _runtime_from_mapping(record)
        _require_stored_fields(
            {key: value for key, value in record.items() if key != "node_id"},
            runtimes[node_id],
        )
    nodes: list[AlgorithmNodeCheckpointPayload] = []
    for node_id, shell in _split_records(
        reader,
        shell_shards,
        total,
        {
            "node_id",
            "parent_node_id",
            "parent_node_order",
            "branch_from_parent",
            "depth",
            "generated_all_branches",
            "state_summary",
        },
    ):
        if node_id not in states or node_id not in runtimes:
            raise ShardedCheckpointManifestError.invalid_input(
                "split_node_ID_coverage_mismatch"
            )
        state_payload = states.pop(node_id)
        runtime = runtimes.pop(node_id)
        if json.dumps(shell["state_summary"], sort_keys=True) != json.dumps(
            state_payload.state_summary, sort_keys=True
        ):
            raise ShardedCheckpointManifestError.invalid_input(
                "conflicting_shell/state_summary"
            )
        nodes.append(
            AlgorithmNodeCheckpointPayload(
                node_id=node_id,
                parent_node_id=optional_int_field(shell, "parent_node_id"),
                branch_from_parent=cast(
                    "CheckpointAtomPayload | None", shell["branch_from_parent"]
                ),
                parent_node_order=cast("list[int] | None", shell["parent_node_order"]),
                depth=require_int_field(shell, "depth"),
                generated_all_branches=require_bool_field(
                    shell, "generated_all_branches"
                ),
                state_payload=state_payload,
                unopened_branches=runtime.unopened_branches,
                linked_children=runtime.linked_children,
                evaluation=runtime.evaluation,
                exploration_index=runtime.exploration_index,
            )
        )
    return nodes


def _algorithm_node_payload_from_mapping(
    payload: dict[str, object],
) -> AlgorithmNodeCheckpointPayload:
    """Build one typed node payload from a C2b JSON record."""
    runtime = _runtime_from_mapping(payload)
    return AlgorithmNodeCheckpointPayload(
        node_id=require_int_field(payload, "node_id"),
        parent_node_id=optional_int_field(payload, "parent_node_id"),
        branch_from_parent=cast(
            "CheckpointAtomPayload | None",
            payload.get("branch_from_parent"),
        ),
        parent_node_order=cast("list[int] | None", payload.get("parent_node_order")),
        depth=require_int_field(payload, "depth"),
        state_payload=_state_payload_from_mapping(
            require_mapping_field(payload, "state_payload")
        ),
        generated_all_branches=require_bool_field(payload, "generated_all_branches"),
        unopened_branches=runtime.unopened_branches,
        linked_children=runtime.linked_children,
        evaluation=runtime.evaluation,
        exploration_index=runtime.exploration_index,
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
        selection_step_count=require_int(
            payload.get("selection_step_count", 0), field_name="selection_step_count"
        ),
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
