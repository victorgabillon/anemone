"""Direct runtime restore from sharded checkpoint files."""

# pylint: disable=duplicate-code,import-outside-toplevel

from __future__ import annotations

# pyright: reportPrivateUsage=false
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, cast

from anemone._valanga_types import AnyTurnState

from .payloads import (
    CheckpointAtomPayload,
    CheckpointNodeStatePayload,
    DeltaCheckpointStatePayload,
    ExplorationIndexCheckpointPayload,
    LinkedChildCheckpointPayload,
    NodeEvaluationCheckpointPayload,
)
from .sharded_decode import (
    _algorithm_node_payload_from_mapping,
    _exploration_index_payload_from_mapping,
    _linked_child_payload_from_mapping,
    _node_evaluation_payload_from_mapping,
    _selector_payload_from_mapping,
    _state_payload_from_mapping,
    _tree_expansions_payload_from_mapping,
)
from .sharded_errors import ShardedCheckpointManifestError
from .sharded_io import (
    _group_manifest_shards,
    _iter_jsonl_shard_record_batches,
    _load_mapping_json_shard,
    _only_manifest_shard,
    _optional_mapping_json_shard,
    _read_jsonl_shard_records,
    _required_manifest_shards,
)
from .sharded_manifest import read_sharded_checkpoint_manifest
from .state_handles import (
    CheckpointPayloadStore,
    DenseCheckpointPayloadStore,
    DictCheckpointPayloadStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from valanga import Dynamics, RepresentationFactory, StateModifications
    from valanga.evaluator_types import EvaluatorInput
    from valanga.policy import NotifyProgressCallable

    from anemone.dynamics import SearchDynamics
    from anemone.factory import SearchArgs
    from anemone.hooks.search_hooks import SearchHooks
    from anemone.node_evaluation.direct.protocols import MasterStateValueEvaluator
    from anemone.node_evaluation.tree.factory import NodeTreeEvaluationFactory
    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
    from anemone.tree_exploration import TreeExploration

    from ._protocols import IncrementalStateCheckpointCodec
    from .sharded_types import ShardedCheckpointShardRef

_DEFAULT_NODE_RUNTIME_RESTORE_BATCH_SIZE = 2048


@dataclass(frozen=True, slots=True)
class _ShardedNodeShell:
    """Compact sharded-restore node metadata retained across restore passes."""

    node_id: int
    parent_node_id: int | None
    branch_from_parent: CheckpointAtomPayload | None
    depth: int
    generated_all_branches: bool
    state_summary: object | None


@dataclass(frozen=True, slots=True)
class _ShardedNodeRuntime:
    """Shard-local runtime payload used after compact node creation."""

    node_id: int
    generated_all_branches: bool
    unopened_branches: list[CheckpointAtomPayload]
    linked_children: list[LinkedChildCheckpointPayload]
    evaluation: NodeEvaluationCheckpointPayload | None
    exploration_index: ExplorationIndexCheckpointPayload | None


def load_search_from_sharded_checkpoint[
    StateT: AnyTurnState,
    ActionT,
](
    checkpoint_dir: str | Path,
    *,
    state_codec: IncrementalStateCheckpointCodec[StateT],
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    args: SearchArgs,
    state_type: type[StateT],
    master_state_value_evaluator: MasterStateValueEvaluator,
    random_generator: Random | None = None,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None = None,
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT] | None = None,
    notify_progress: NotifyProgressCallable | None = None,
    hooks: SearchHooks | None = None,
    restore_memory_phase_logger: Callable[[str, Mapping[str, object]], None]
    | None = None,
) -> TreeExploration[AlgorithmNode[StateT]]:
    """Experimentally restore runtime directly from C2b shards.

    This C2d path is opt-in and intentionally keeps the C2c typed-payload reader
    unchanged. It retains compact per-node shells and a direct state payload
    store, then restores full node runtime data one shard at a time.
    """
    from anemone._runtime_assembly import assemble_search_runtime_dependencies
    from anemone.node_evaluation.tree.factory import NodeTreeMinmaxEvaluationFactory
    from anemone.progress_monitor.progress_monitor import create_stopping_criterion
    from anemone.tree_exploration import TreeExploration
    from anemone.tree_exploration_debug import (
        log_final_best_line,
        maybe_log_iteration_progress,
    )

    from .load import (
        _build_tree_from_node_shells,
        _CheckpointNodeRuntime,
        _CheckpointNodeShell,
        _create_nodes_from_node_shells,
        _create_state_handles_from_node_ids,
        _create_state_resolver_from_payload_store,
        _link_nodes,
        _restore_explicit_selector_state,
        _restore_inferred_depth_selector_state,
        _restore_node_runtime_state,
        _restore_runtime_metadata,
        _restore_tree_expansions_runtime_state,
        _validate_checkpoint_node_metadata,
    )

    resolved_checkpoint_dir = Path(checkpoint_dir)
    manifest = read_sharded_checkpoint_manifest(
        resolved_checkpoint_dir / "manifest.json"
    )
    grouped_shards = _group_manifest_shards(manifest)
    metadata = _load_mapping_json_shard(
        resolved_checkpoint_dir,
        _only_manifest_shard(grouped_shards, "metadata"),
    )
    node_record_shards = grouped_shards.get("node_records", [])
    split_layout = not node_record_shards and bool(grouped_shards.get("node_shells"))
    common_metadata = {
        "checkpoint_dir": str(resolved_checkpoint_dir),
        "node_count": manifest.total_node_count,
        "root_node_id": metadata.get("root_node_id"),
        "format_version": metadata.get("format_version"),
        "sharded_layout": "split" if split_layout else "node_records",
    }
    _log_sharded_restore_memory_phase(
        restore_memory_phase_logger,
        "after_manifest_load",
        **common_metadata,
        shard_count=len(manifest.shards),
    )

    if split_layout:
        (
            node_shells,
            state_payload_store,
            branch_count,
            anchor_count,
            delta_count,
            delta_parent_node_ids,
        ) = _load_split_incremental_node_shells(
            resolved_checkpoint_dir,
            _required_manifest_shards(grouped_shards, "node_shells"),
            _required_manifest_shards(grouped_shards, "state_payloads"),
        )
    else:
        (
            node_shells,
            state_payload_store,
            branch_count,
            anchor_count,
            delta_count,
            delta_parent_node_ids,
        ) = _load_incremental_node_shells(
            resolved_checkpoint_dir,
            _required_manifest_shards(grouped_shards, "node_records"),
        )
    if len(node_shells) != manifest.total_node_count:
        raise ShardedCheckpointManifestError.node_count_mismatch()
    if cast("int | None", metadata.get("node_count")) not in {
        None,
        manifest.total_node_count,
    }:
        raise ShardedCheckpointManifestError.node_count_mismatch()
    if split_layout:
        branch_count = (
            manifest.total_branch_count
            if manifest.total_branch_count is not None
            else cast("int", metadata.get("branch_count", 0))
        )
    if (
        not split_layout
        and manifest.total_branch_count is not None
        and branch_count != (manifest.total_branch_count)
    ):
        raise ShardedCheckpointManifestError.branch_count_mismatch()

    _validate_checkpoint_node_metadata(
        format_version=cast("int", metadata["format_version"]),
        root_node_id=cast("int", metadata["root_node_id"]),
        node_ids=[node_shell.node_id for node_shell in node_shells],
        parent_node_ids=[node_shell.parent_node_id for node_shell in node_shells],
        delta_parent_node_ids=delta_parent_node_ids,
    )
    checkpoint_random_generator = random_generator if random_generator else Random()
    dependencies = assemble_search_runtime_dependencies(
        dynamics=dynamics,
        args=args,
        random_generator=checkpoint_random_generator,
        master_state_evaluator=master_state_value_evaluator,
        state_representation_factory=state_representation_factory,
        node_tree_evaluation_factory=(
            node_tree_evaluation_factory or NodeTreeMinmaxEvaluationFactory()
        ),
        hooks=hooks,
    )

    state_resolver = _create_state_resolver_from_payload_store(
        payload_store=state_payload_store,
        state_codec=state_codec,
    )
    _log_sharded_restore_memory_phase(
        restore_memory_phase_logger,
        "after_state_payload_store_built",
        **common_metadata,
        anchor_count=anchor_count,
        delta_count=delta_count,
        shell_count=len(node_shells),
        state_payload_count=len(state_payload_store),
        node_shell_type="compact",
        retained_full_node_payloads=False,
    )
    state_handles_by_id = _create_state_handles_from_node_ids(
        node_ids=[node_shell.node_id for node_shell in node_shells],
        state_resolver=state_resolver,
    )
    _ = state_type
    typed_node_shells = cast("list[_CheckpointNodeShell]", node_shells)
    nodes_by_id = _create_nodes_from_node_shells(
        node_factory=dependencies.tree_manager.tree_manager.node_factory,
        node_shells=typed_node_shells,
        state_handles_by_id=state_handles_by_id,
    )
    restored_tree = _build_tree_from_node_shells(
        root_node_id=cast("int", metadata["root_node_id"]),
        node_shells=cast("Sequence[_CheckpointNodeShell]", typed_node_shells),
        nodes_by_id=nodes_by_id,
        branch_count=(
            manifest.total_branch_count
            if manifest.total_branch_count is not None
            else branch_count
        ),
    )
    _log_sharded_restore_memory_phase(
        restore_memory_phase_logger,
        "after_nodes_created",
        **common_metadata,
        shell_count=len(node_shells),
        state_payload_count=len(state_payload_store),
        node_shell_type="compact",
        retained_full_node_payloads=False,
    )

    if split_layout:
        shell_by_node_id = {
            node_shell.node_id: node_shell for node_shell in node_shells
        }
        restored_branch_count = 0
        node_runtime_batch_count = 0
        node_runtime_record_count = 0
        node_runtime_zstd_streaming = False
        for shard in _required_manifest_shards(grouped_shards, "node_runtime"):
            if shard.encoding == "jsonl_zst":
                node_runtime_zstd_streaming = True
            for record_batch in _iter_jsonl_shard_record_batches(
                resolved_checkpoint_dir,
                shard,
                batch_size=_DEFAULT_NODE_RUNTIME_RESTORE_BATCH_SIZE,
            ):
                node_runtimes = [
                    _sharded_node_runtime_from_mapping(
                        record,
                        node_shell=shell_by_node_id[cast("int", record["node_id"])],
                    )
                    for record in record_batch
                ]
                node_runtime_batch_count += 1
                node_runtime_record_count += len(node_runtimes)
                restored_branch_count += sum(
                    len(node_runtime.linked_children) for node_runtime in node_runtimes
                )
                typed_node_runtimes = cast(
                    "list[_CheckpointNodeRuntime]",
                    node_runtimes,
                )
                _link_nodes(typed_node_runtimes, nodes_by_id=nodes_by_id)
                _restore_node_runtime_state(
                    typed_node_runtimes,
                    nodes_by_id=nodes_by_id,
                )
                del node_runtimes
                del record_batch
        if manifest.total_branch_count is not None and restored_branch_count != (
            manifest.total_branch_count
        ):
            raise ShardedCheckpointManifestError.branch_count_mismatch()
        branch_count = restored_branch_count
    else:
        node_runtime_batch_count = None
        node_runtime_record_count = None
        node_runtime_zstd_streaming = None
        for shard in _required_manifest_shards(grouped_shards, "node_records"):
            node_payloads = [
                _algorithm_node_payload_from_mapping(record)
                for record in _read_jsonl_shard_records(resolved_checkpoint_dir, shard)
            ]
            _link_nodes(node_payloads, nodes_by_id=nodes_by_id)
            _restore_node_runtime_state(node_payloads, nodes_by_id=nodes_by_id)
    _log_sharded_restore_memory_phase(
        restore_memory_phase_logger,
        "after_edges_and_runtime_state_restored",
        **common_metadata,
        branch_count=branch_count,
        shell_count=len(node_shells),
        state_payload_count=len(state_payload_store),
        node_shell_type="compact",
        retained_full_node_payloads=False,
        node_runtime_batch_size=(
            _DEFAULT_NODE_RUNTIME_RESTORE_BATCH_SIZE if split_layout else None
        ),
        node_runtime_batch_count=node_runtime_batch_count,
        node_runtime_record_count=node_runtime_record_count,
        node_runtime_streaming=split_layout,
        node_runtime_zstd_streaming=node_runtime_zstd_streaming,
    )

    node_selector = dependencies.node_selector_create()
    runtime = TreeExploration(
        tree=restored_tree,
        tree_manager=dependencies.tree_manager,
        stopping_criterion=create_stopping_criterion(
            args=args.stopping_criterion,
            node_selector=node_selector,
        ),
        node_selector=node_selector,
        recommend_branch_after_exploration=args.recommender_rule,
        notify_percent_function=notify_progress,
        iteration_progress_reporter=maybe_log_iteration_progress,
        search_result_reporter=log_final_best_line,
    )
    _restore_runtime_metadata(
        runtime=runtime,
        evaluator_version=cast("int", metadata["evaluator_version"]),
        rng_state=metadata.get("rng_state"),
        random_generator=checkpoint_random_generator,
    )
    latest_tree_expansions = _optional_mapping_json_shard(
        resolved_checkpoint_dir,
        grouped_shards,
        "latest_expansions",
    )
    latest_tree_expansions_payload = (
        None
        if latest_tree_expansions is None
        else _tree_expansions_payload_from_mapping(latest_tree_expansions)
    )
    selector_state = _optional_mapping_json_shard(
        resolved_checkpoint_dir,
        grouped_shards,
        "selector",
    )
    selector_payload = (
        None
        if selector_state is None
        else _selector_payload_from_mapping(selector_state)
    )
    _restore_tree_expansions_runtime_state(
        runtime=runtime,
        payload=latest_tree_expansions_payload,
        nodes_by_id=nodes_by_id,
    )
    _restore_explicit_selector_state(runtime=runtime, payload=selector_payload)
    _restore_inferred_depth_selector_state(
        runtime=runtime,
        payload=latest_tree_expansions_payload,
        nodes_by_id=nodes_by_id,
    )
    _log_sharded_restore_memory_phase(
        restore_memory_phase_logger,
        "after_selector_restored",
        **common_metadata,
    )
    _log_sharded_restore_memory_phase(
        restore_memory_phase_logger,
        "after_incremental_restore_done",
        **common_metadata,
        branch_count=runtime.tree.branch_count,
    )
    return runtime


def _load_incremental_node_shells(
    checkpoint_dir: Path,
    node_shards: list[ShardedCheckpointShardRef],
) -> tuple[
    list[_ShardedNodeShell],
    CheckpointPayloadStore,
    int,
    int,
    int,
    list[tuple[int, int]],
]:
    """Load compact node shells and state payloads for incremental restoration."""
    node_shells: list[_ShardedNodeShell] = []
    state_payloads_by_node_id: dict[int, CheckpointNodeStatePayload] = {}
    delta_parent_node_ids: list[tuple[int, int]] = []
    branch_count = 0
    anchor_count = 0
    delta_count = 0
    for shard in node_shards:
        for record in _read_jsonl_shard_records(checkpoint_dir, shard):
            node_id = cast("int", record["node_id"])
            state_payload = _state_payload_from_mapping(
                cast("dict[str, object]", record["state_payload"])
            )
            node_shell = _sharded_node_shell_from_mapping(
                record,
                state_summary=state_payload.state_summary,
            )
            node_shells.append(node_shell)
            state_payloads_by_node_id[node_id] = state_payload
            branch_count += len(cast("list[object]", record.get("linked_children", [])))
            if isinstance(state_payload, DeltaCheckpointStatePayload):
                delta_count += 1
                delta_parent_node_ids.append(
                    (node_id, state_payload.state_parent_node_id)
                )
            else:
                anchor_count += 1
    return (
        node_shells,
        _build_sharded_payload_store(state_payloads_by_node_id),
        branch_count,
        anchor_count,
        delta_count,
        delta_parent_node_ids,
    )


def _load_split_incremental_node_shells(
    checkpoint_dir: Path,
    node_shell_shards: list[ShardedCheckpointShardRef],
    state_payload_shards: list[ShardedCheckpointShardRef],
) -> tuple[
    list[_ShardedNodeShell],
    CheckpointPayloadStore,
    int,
    int,
    int,
    list[tuple[int, int]],
]:
    """Load split-layout shells and state payloads without reading runtime shards."""
    state_payloads_by_node_id: dict[int, CheckpointNodeStatePayload] = {}
    state_summary_by_node_id: dict[int, object | None] = {}
    delta_parent_node_ids: list[tuple[int, int]] = []
    anchor_count = 0
    delta_count = 0
    for shard in state_payload_shards:
        for record in _read_jsonl_shard_records(checkpoint_dir, shard):
            node_id = cast("int", record["node_id"])
            state_payload = _state_payload_from_mapping(
                cast("dict[str, object]", record["state_payload"])
            )
            state_payloads_by_node_id[node_id] = state_payload
            state_summary_by_node_id[node_id] = state_payload.state_summary
            if isinstance(state_payload, DeltaCheckpointStatePayload):
                delta_count += 1
                delta_parent_node_ids.append(
                    (node_id, state_payload.state_parent_node_id)
                )
            else:
                anchor_count += 1

    node_shells: list[_ShardedNodeShell] = []
    for shard in node_shell_shards:
        for record in _read_jsonl_shard_records(checkpoint_dir, shard):
            node_id = cast("int", record["node_id"])
            node_shells.append(
                _sharded_node_shell_from_mapping(
                    record,
                    state_summary=state_summary_by_node_id.get(node_id),
                )
            )

    return (
        node_shells,
        _build_sharded_payload_store(state_payloads_by_node_id),
        0,
        anchor_count,
        delta_count,
        delta_parent_node_ids,
    )


def _build_sharded_payload_store(
    state_payloads_by_node_id: dict[int, CheckpointNodeStatePayload],
) -> CheckpointPayloadStore:
    """Build the cheapest payload store for sharded incremental restore."""
    if _node_ids_are_dense_zero_based(state_payloads_by_node_id):
        return DenseCheckpointPayloadStore(
            [
                state_payloads_by_node_id[node_id]
                for node_id in range(len(state_payloads_by_node_id))
            ]
        )
    return DictCheckpointPayloadStore(state_payloads_by_node_id)


def _node_ids_are_dense_zero_based(node_ids: Iterable[int]) -> bool:
    """Return whether node ids cover the exact range ``0..N-1``."""
    node_ids = list(node_ids)
    if not node_ids:
        return False
    return set(node_ids) == set(range(len(node_ids)))


def _sharded_node_shell_from_mapping(
    payload: dict[str, object],
    *,
    state_summary: object | None,
) -> _ShardedNodeShell:
    """Build compact node metadata without retaining full state/runtime payloads."""
    return _ShardedNodeShell(
        node_id=cast("int", payload["node_id"]),
        parent_node_id=cast("int | None", payload["parent_node_id"]),
        branch_from_parent=cast(
            "CheckpointAtomPayload | None",
            payload["branch_from_parent"],
        ),
        depth=cast("int", payload["depth"]),
        generated_all_branches=cast("bool", payload["generated_all_branches"]),
        state_summary=state_summary,
    )


def _sharded_node_runtime_from_mapping(
    payload: dict[str, object],
    *,
    node_shell: _ShardedNodeShell,
) -> _ShardedNodeRuntime:
    """Build one shard-local runtime payload from split node-runtime JSON."""
    return _ShardedNodeRuntime(
        node_id=cast("int", payload["node_id"]),
        generated_all_branches=node_shell.generated_all_branches,
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


def _log_sharded_restore_memory_phase(
    restore_memory_phase_logger: Callable[[str, Mapping[str, object]], None] | None,
    phase_name: str,
    **metadata: object,
) -> None:
    """Emit one optional sharded restore-memory phase marker."""
    if restore_memory_phase_logger is None:
        return
    restore_memory_phase_logger(phase_name, metadata)
