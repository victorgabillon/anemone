"""Restore-side checkpoint state resolver and lazy handle construction."""

# pyright: reportUnusedFunction=false

# ruff: noqa: TC001,TC003

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from anemone._valanga_types import AnyTurnState

from .payloads import (
    AlgorithmNodeCheckpointPayload,
    CheckpointNodeStatePayload,
    SearchRuntimeCheckpointPayload,
)
from .state_handles import (
    CheckpointBackedStateHandle,
    CheckpointPayloadStore,
    CheckpointStateResolver,
    DenseCheckpointPayloadStore,
    DictCheckpointPayloadStore,
)

if TYPE_CHECKING:
    from anemone.nodes.state_handles import StateHandle

    from ._protocols import IncrementalStateCheckpointCodec


def _create_state_resolver[
    StateT: AnyTurnState,
](
    *,
    payload: SearchRuntimeCheckpointPayload,
    state_codec: IncrementalStateCheckpointCodec[StateT],
) -> CheckpointStateResolver[StateT]:
    """Create the shared lazy resolver for all checkpoint-backed state handles."""
    return _create_state_resolver_from_node_payloads(
        node_payloads=payload.tree.nodes,
        state_codec=state_codec,
    )


def _create_state_resolver_from_node_payloads[
    StateT: AnyTurnState,
](
    *,
    node_payloads: Iterable[AlgorithmNodeCheckpointPayload],
    state_codec: IncrementalStateCheckpointCodec[StateT],
) -> CheckpointStateResolver[StateT]:
    """Create the shared lazy resolver from checkpoint node payloads."""
    payload_store = _build_checkpoint_payload_store(node_payloads)
    return _create_state_resolver_from_payload_store(
        payload_store=payload_store,
        state_codec=state_codec,
    )


def _create_state_resolver_from_payload_store[
    StateT: AnyTurnState,
](
    *,
    payload_store: CheckpointPayloadStore,
    state_codec: IncrementalStateCheckpointCodec[StateT],
) -> CheckpointStateResolver[StateT]:
    """Create the shared lazy resolver from a prebuilt checkpoint payload store."""
    return CheckpointStateResolver(
        state_codec=state_codec,
        state_payloads_by_node_id=payload_store,
    )


def _build_checkpoint_payload_store(
    node_payloads: Iterable[AlgorithmNodeCheckpointPayload],
) -> DenseCheckpointPayloadStore | DictCheckpointPayloadStore:
    """Build the cheapest in-memory payload store for one restored checkpoint."""
    payloads = tuple(node_payloads)
    if _node_payload_ids_are_dense_zero_based(payloads):
        dense_payloads: list[CheckpointNodeStatePayload] = [
            payload.state_payload
            for payload in sorted(payloads, key=lambda item: item.node_id)
        ]
        return DenseCheckpointPayloadStore(dense_payloads)
    return DictCheckpointPayloadStore(
        {node_payload.node_id: node_payload.state_payload for node_payload in payloads}
    )


def _node_payload_ids_are_dense_zero_based(
    node_payloads: Iterable[AlgorithmNodeCheckpointPayload],
) -> bool:
    """Return whether checkpoint node ids cover the exact range ``0..N-1``."""
    payloads = tuple(node_payloads)
    if not payloads:
        return False
    return _node_ids_are_dense_zero_based(payload.node_id for payload in payloads)


def _node_ids_are_dense_zero_based(node_ids: Iterable[int]) -> bool:
    """Return whether checkpoint node ids cover the exact range ``0..N-1``."""
    node_ids = list(node_ids)
    if not node_ids:
        return False
    expected_ids = set(range(len(node_ids)))
    return set(node_ids) == expected_ids


def _create_state_handles[
    StateT: AnyTurnState,
](
    *,
    payload: SearchRuntimeCheckpointPayload,
    state_resolver: CheckpointStateResolver[StateT],
) -> dict[int, StateHandle[StateT]]:
    """Create one lazy checkpoint-backed handle for each restored node."""
    return _create_state_handles_from_node_payloads(
        node_payloads=payload.tree.nodes,
        state_resolver=state_resolver,
    )


def _create_state_handles_from_node_payloads[
    StateT: AnyTurnState,
](
    *,
    node_payloads: Iterable[AlgorithmNodeCheckpointPayload],
    state_resolver: CheckpointStateResolver[StateT],
) -> dict[int, StateHandle[StateT]]:
    """Create one lazy checkpoint-backed handle for each restored node payload."""
    return {
        node_payload.node_id: CheckpointBackedStateHandle(
            resolver=state_resolver,
            node_id=node_payload.node_id,
        )
        for node_payload in node_payloads
    }


def _create_state_handles_from_node_ids[
    StateT: AnyTurnState,
](
    *,
    node_ids: Iterable[int],
    state_resolver: CheckpointStateResolver[StateT],
) -> dict[int, StateHandle[StateT]]:
    """Create one lazy checkpoint-backed handle for each restored node id."""
    return {
        node_id: CheckpointBackedStateHandle(
            resolver=state_resolver,
            node_id=node_id,
        )
        for node_id in node_ids
    }
