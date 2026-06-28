"""Checkpoint restore metadata, validation, and runtime metadata helpers."""

# pyright: reportUnusedFunction=false

# ruff: noqa: TC001,TC003

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from random import Random
from time import perf_counter
from typing import Any, cast

from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.tree_exploration import TreeExploration
from anemone.utils.logger import checkpoint_logger

from .payloads import (
    CHECKPOINT_FORMAT_VERSION,
    AlgorithmNodeCheckpointPayload,
    DeltaCheckpointStatePayload,
    SearchRuntimeCheckpointPayload,
)

type RestoreMemoryPhaseLogger = Callable[[str, Mapping[str, object]], None]


class CheckpointRestoreError(ValueError):
    """Raised when a checkpoint payload cannot be restored safely."""

    @classmethod
    def unsupported_format_version(cls, actual: int) -> CheckpointRestoreError:
        """Return the error for an unsupported checkpoint format version."""
        return cls(
            "Unsupported checkpoint format version "
            f"{actual}; expected {CHECKPOINT_FORMAT_VERSION}."
        )

    @classmethod
    def empty_tree(cls) -> CheckpointRestoreError:
        """Return the error for an empty tree checkpoint."""
        return cls("Cannot restore a checkpoint with no nodes.")

    @classmethod
    def duplicate_node_ids(cls) -> CheckpointRestoreError:
        """Return the error for duplicate node ids."""
        return cls("Checkpoint contains duplicate node ids.")

    @classmethod
    def missing_root(cls, root_node_id: int) -> CheckpointRestoreError:
        """Return the error for a root id that has no node payload."""
        return cls(f"Root node id {root_node_id} is not present in nodes.")

    @classmethod
    def invalid_exploration_index_payload(cls) -> CheckpointRestoreError:
        """Return the error for unsupported exploration-index payload shape."""
        return cls("Exploration index payloads must be dicts produced by the exporter.")

    @classmethod
    def live_tree_node_in_index_payload(cls) -> CheckpointRestoreError:
        """Return the error for accidentally serialized live tree-node pointers."""
        return cls("Exploration index payload unexpectedly contains live tree_node.")

    @classmethod
    def unsupported_exploration_index_kind(
        cls,
        kind: str | None,
    ) -> CheckpointRestoreError:
        """Return the error for an unknown exploration-index kind."""
        return cls(f"Unsupported exploration index kind {kind!r}.")

    @classmethod
    def unknown_node_id(cls, node_id: int) -> CheckpointRestoreError:
        """Return the error for a node reference that cannot be resolved."""
        return cls(f"Checkpoint references unknown node id {node_id}.")

    @classmethod
    def delta_node_missing_parent(cls, node_id: int) -> CheckpointRestoreError:
        """Return the error for a delta payload that lacks a parent node id."""
        return cls(f"Delta checkpoint node {node_id} must reference a parent node id.")


@contextmanager
def _log_restore_phase(phase_name: str, **metadata: object) -> Any:
    """Log one checkpoint-restore phase start and completion timing."""
    _log_checkpoint_restore_event(phase_name, "start", **metadata)
    started_at = perf_counter()
    try:
        yield
    finally:
        _log_checkpoint_restore_event(
            phase_name,
            "done",
            elapsed_s=round(perf_counter() - started_at, 6),
            **metadata,
        )


def _log_checkpoint_restore_event(
    phase_name: str,
    status: str,
    **metadata: object,
) -> None:
    """Emit one structured checkpoint-restore log line."""
    metadata_suffix = " ".join(
        f"{key}={value}" for key, value in metadata.items() if value is not None
    )
    message = f"[checkpoint-restore] phase={phase_name} status={status}"
    if metadata_suffix:
        message = f"{message} {metadata_suffix}"
    checkpoint_logger.debug(message)


def _log_restore_memory_phase(
    restore_memory_phase_logger: RestoreMemoryPhaseLogger | None,
    phase_name: str,
    **metadata: object,
) -> None:
    """Emit one optional restore-memory phase marker."""
    if restore_memory_phase_logger is None:
        return
    restore_memory_phase_logger(phase_name, metadata)


def _checkpoint_state_payload_counts(
    payload: SearchRuntimeCheckpointPayload,
) -> tuple[int, int]:
    """Return the anchor and delta payload counts for one checkpoint."""
    anchor_count = 0
    delta_count = 0
    for node_payload in payload.tree.nodes:
        if isinstance(node_payload.state_payload, DeltaCheckpointStatePayload):
            delta_count += 1
        else:
            anchor_count += 1
    return anchor_count, delta_count


def _latest_expansion_count(payload: SearchRuntimeCheckpointPayload) -> int:
    """Return the number of latest expansion records stored in the payload."""
    latest_tree_expansions = payload.latest_tree_expansions
    if latest_tree_expansions is None:
        return 0
    return len(latest_tree_expansions.expansions_with_node_creation) + len(
        latest_tree_expansions.expansions_without_node_creation
    )


def _validate_payload(payload: SearchRuntimeCheckpointPayload) -> None:
    """Validate the minimal structural facts needed before restoring."""
    _validate_checkpoint_node_payloads(
        format_version=payload.format_version,
        root_node_id=payload.tree.root_node_id,
        node_payloads=payload.tree.nodes,
    )


def _validate_checkpoint_node_payloads(
    *,
    format_version: int,
    root_node_id: int,
    node_payloads: Sequence[AlgorithmNodeCheckpointPayload],
) -> None:
    """Validate structural facts needed by any checkpoint restore path."""
    _validate_checkpoint_node_metadata(
        format_version=format_version,
        root_node_id=root_node_id,
        node_ids=[node_payload.node_id for node_payload in node_payloads],
        parent_node_ids=[node_payload.parent_node_id for node_payload in node_payloads],
        delta_parent_node_ids=[
            (
                node_payload.node_id,
                node_payload.state_payload.state_parent_node_id,
            )
            for node_payload in node_payloads
            if isinstance(node_payload.state_payload, DeltaCheckpointStatePayload)
        ],
    )


def _validate_checkpoint_node_metadata(
    *,
    format_version: int,
    root_node_id: int,
    node_ids: Sequence[int],
    parent_node_ids: Sequence[int | None],
    delta_parent_node_ids: Sequence[tuple[int, int]],
) -> None:
    """Validate structural facts shared by monolithic and sharded restore paths."""
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointRestoreError.unsupported_format_version(format_version)
    if not node_ids:
        raise CheckpointRestoreError.empty_tree()

    node_id_set = set(node_ids)
    if len(node_ids) != len(node_id_set):
        raise CheckpointRestoreError.duplicate_node_ids()
    if root_node_id not in node_id_set:
        raise CheckpointRestoreError.missing_root(root_node_id)
    for parent_node_id in parent_node_ids:
        if parent_node_id is not None and parent_node_id not in node_id_set:
            raise CheckpointRestoreError.unknown_node_id(parent_node_id)
    for node_id, state_parent_node_id in delta_parent_node_ids:
        if state_parent_node_id == node_id:
            raise CheckpointRestoreError.delta_node_missing_parent(node_id)
        if state_parent_node_id not in node_id_set:
            raise CheckpointRestoreError.unknown_node_id(state_parent_node_id)


def _restore_runtime_state(
    *,
    runtime: TreeExploration[AlgorithmNode[Any]],
    payload: SearchRuntimeCheckpointPayload,
    random_generator: Random,
) -> None:
    """Restore runtime-level counters, evaluator version, and RNG state."""
    _restore_runtime_metadata(
        runtime=runtime,
        evaluator_version=payload.evaluator_version,
        rng_state=payload.rng_state,
        random_generator=random_generator,
    )


def _restore_runtime_metadata(
    *,
    runtime: TreeExploration[AlgorithmNode[Any]],
    evaluator_version: int,
    rng_state: object | None,
    random_generator: Random,
) -> None:
    """Restore runtime-level counters, evaluator version, and RNG state."""
    runtime.evaluator_version = evaluator_version
    node_evaluator = runtime.tree_manager.node_evaluator
    if node_evaluator is not None:
        node_evaluator.current_evaluator_version = evaluator_version

    if rng_state is not None:
        random_generator.setstate(cast("tuple[Any, ...]", rng_state))
