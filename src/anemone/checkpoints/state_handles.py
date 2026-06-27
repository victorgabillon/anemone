"""Checkpoint-backed lazy state handles used during restore."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast

from valanga import State

from ._protocols import CheckpointStateSummary, IncrementalStateCheckpointCodec
from .payloads import (
    AnchorCheckpointStatePayload,
    CheckpointNodeStatePayload,
    DeltaCheckpointStatePayload,
)
from .value_serialization import deserialize_checkpoint_atom


class CheckpointStateResolutionError(KeyError):
    """Raised when one checkpoint-backed state cannot be resolved safely."""

    @classmethod
    def delta_node_missing_parent(
        cls, node_id: int
    ) -> "CheckpointStateResolutionError":
        """Return the error for a delta node that lacks a parent node id."""
        return cls(f"Delta checkpoint node {node_id} has no parent node id.")

    @classmethod
    def payload_chain_cycle(cls, node_id: int) -> "CheckpointStateResolutionError":
        """Return the error for a cyclic checkpoint payload chain."""
        return cls(f"Cycle in checkpoint state payload chain at node {node_id}.")


class CheckpointPayloadStore(Protocol):
    """Mapping-like store used by lazy checkpoint state resolvers."""

    def __getitem__(self, node_id: int) -> CheckpointNodeStatePayload: ...

    def __iter__(self) -> Iterator[int]: ...

    def __len__(self) -> int: ...

    def get(
        self,
        node_id: int,
        default: CheckpointNodeStatePayload | None = None,
    ) -> CheckpointNodeStatePayload | None: ...

    def values(self) -> object: ...

    def items(self) -> object: ...


@dataclass(frozen=True, slots=True)
class DictCheckpointPayloadStore(Mapping[int, CheckpointNodeStatePayload]):
    """Thin mapping wrapper preserving legacy dict-backed checkpoint payloads."""

    payloads_by_node_id: dict[int, CheckpointNodeStatePayload]

    def __getitem__(self, node_id: int) -> CheckpointNodeStatePayload:
        return self.payloads_by_node_id[node_id]

    def __iter__(self) -> Iterator[int]:
        return iter(self.payloads_by_node_id)

    def __len__(self) -> int:
        return len(self.payloads_by_node_id)


@dataclass(frozen=True, slots=True)
class DenseCheckpointPayloadStore(Mapping[int, CheckpointNodeStatePayload]):
    """Dense zero-based payload store indexed directly by checkpoint node id."""

    payloads_by_dense_node_id: list[CheckpointNodeStatePayload]

    def __getitem__(self, node_id: int) -> CheckpointNodeStatePayload:
        if node_id < 0 or node_id >= len(self.payloads_by_dense_node_id):
            raise KeyError(node_id)
        return self.payloads_by_dense_node_id[node_id]

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.payloads_by_dense_node_id)))

    def __len__(self) -> int:
        return len(self.payloads_by_dense_node_id)


@dataclass(slots=True)
class CheckpointStateResolver[StateT: State = State]:
    """Shared lazy state resolver used during one checkpoint restore session.

    The resolver owns the restore-session decode cache. Multiple thin
    checkpoint-backed handles for the same node id intentionally share this
    resolver so the underlying checkpoint anchor/delta chain is decoded only
    once per node.
    """

    state_codec: IncrementalStateCheckpointCodec[StateT]
    state_payloads_by_node_id: CheckpointPayloadStore | Mapping[int, CheckpointNodeStatePayload]
    _resolved_states: dict[int, StateT] = field(
        default_factory=lambda: cast("dict[int, StateT]", {}),
        init=False,
    )
    _resolving_node_ids: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        """Normalize legacy mappings into an explicit payload-store abstraction."""
        payload_store = self.state_payloads_by_node_id
        if isinstance(
            payload_store,
            DenseCheckpointPayloadStore | DictCheckpointPayloadStore,
        ):
            return
        object.__setattr__(
            self,
            "state_payloads_by_node_id",
            DictCheckpointPayloadStore(dict(payload_store)),
        )

    def payload_for_node_id(self, node_id: int) -> CheckpointNodeStatePayload:
        """Return the raw checkpoint payload for one restored node id."""
        return self.state_payloads_by_node_id[node_id]

    def payload_for_node_id_or_none(
        self, node_id: int
    ) -> CheckpointNodeStatePayload | None:
        """Return the raw checkpoint payload for one node id, if present."""
        try:
            return self.payload_for_node_id(node_id)
        except KeyError:
            return None

    def resolve(self, node_id: int) -> StateT:
        """Return one checkpointed state, decoding it only once."""
        resolved_state = self._resolved_states.get(node_id)
        if resolved_state is not None:
            return resolved_state

        if node_id in self._resolving_node_ids:
            raise CheckpointStateResolutionError.payload_chain_cycle(node_id)
        self._resolving_node_ids.add(node_id)
        try:
            state_payload = self.payload_for_node_id(node_id)
            if isinstance(state_payload, AnchorCheckpointStatePayload):
                resolved_state = self._resolve_anchor(state_payload)
            else:
                resolved_state = self._resolve_delta(
                    node_id=node_id,
                    state_payload=state_payload,
                )
        finally:
            self._resolving_node_ids.discard(node_id)
        self._resolved_states[node_id] = resolved_state
        return resolved_state

    def summary(self, node_id: int) -> CheckpointStateSummary | None:
        """Return optional checkpoint summary metadata for ``node_id``."""
        state_payload = self.payload_for_node_id(node_id)
        return state_payload.state_summary

    def _resolve_anchor(self, state_payload: AnchorCheckpointStatePayload) -> StateT:
        """Resolve one anchor payload into a concrete state."""
        return self.state_codec.load_anchor_ref(state_payload.anchor_ref)

    def _resolve_delta(
        self,
        *,
        node_id: int,
        state_payload: DeltaCheckpointStatePayload,
    ) -> StateT:
        """Resolve one delta payload by first resolving its stored state parent."""
        parent_node_id = state_payload.state_parent_node_id
        if parent_node_id == node_id:
            raise CheckpointStateResolutionError.delta_node_missing_parent(node_id)

        parent_state = self.resolve(parent_node_id)
        return self.state_codec.load_child_from_delta(
            parent_state=parent_state,
            delta_ref=state_payload.delta_ref,
            branch_from_parent=cast(
                "object | None",
                deserialize_checkpoint_atom(state_payload.state_parent_branch),
            ),
        )


@dataclass(frozen=True, slots=True)
class CheckpointBackedStateHandle[StateT: State = State]:
    """Thin lazy handle that resolves one checkpointed node on first access."""

    resolver: CheckpointStateResolver[StateT]
    node_id: int

    def get(self) -> StateT:
        """Resolve and return the concrete state for this checkpoint node."""
        return self.resolver.resolve(self.node_id)

    def checkpoint_payload_for_reuse_or_none(
        self,
    ) -> AnchorCheckpointStatePayload | DeltaCheckpointStatePayload | None:
        """Return the original checkpoint payload when this handle still owns one."""
        payload = payload_for_node_id_or_none(self.resolver, self.node_id)
        if isinstance(
            payload,
            AnchorCheckpointStatePayload | DeltaCheckpointStatePayload,
        ):
            return payload
        return None


def checkpoint_payload_for_reuse_or_none(
    handle: object,
) -> AnchorCheckpointStatePayload | DeltaCheckpointStatePayload | None:
    """Return one reusable checkpoint payload for a lazy checkpoint handle."""
    if not isinstance(handle, CheckpointBackedStateHandle):
        return None
    return handle.checkpoint_payload_for_reuse_or_none()


def payload_for_node_id_or_none(
    resolver: object,
    node_id: int,
) -> object | None:
    """Return a raw checkpoint payload without resolving concrete state."""
    optional_lookup = getattr(resolver, "payload_for_node_id_or_none", None)
    if callable(optional_lookup):
        try:
            return optional_lookup(node_id)
        except KeyError:
            return None

    strict_lookup = getattr(resolver, "payload_for_node_id", None)
    if callable(strict_lookup):
        try:
            return strict_lookup(node_id)
        except KeyError:
            return None

    payloads_by_node_id = getattr(resolver, "state_payloads_by_node_id", None)
    if isinstance(payloads_by_node_id, Mapping):
        return payloads_by_node_id.get(node_id)
    return None


__all__ = [
    "CheckpointBackedStateHandle",
    "CheckpointPayloadStore",
    "CheckpointStateResolutionError",
    "CheckpointStateResolver",
    "DenseCheckpointPayloadStore",
    "DictCheckpointPayloadStore",
    "checkpoint_payload_for_reuse_or_none",
    "payload_for_node_id_or_none",
]
