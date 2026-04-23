"""Explicit state-handle abstractions used by core node types."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from valanga import State

from anemone.checkpoints._protocols import (
    CheckpointStateSummary,
    IncrementalStateCheckpointCodec,
)
from anemone.checkpoints.payloads import (
    AnchorCheckpointStatePayload,
    CheckpointNodeStatePayload,
    DeltaCheckpointStatePayload,
)


class StateHandle[StateT: State = State](Protocol):
    """Minimal explicit handle for retrieving one node state."""

    def get(self) -> StateT:
        """Return the concrete state behind this handle."""
        ...


@dataclass(frozen=True, slots=True)
class MaterializedStateHandle[StateT: State = State]:
    """Handle that simply stores an already materialized state object."""

    state_: StateT

    def get(self) -> StateT:
        """Return the stored concrete state."""
        return self.state_


@dataclass(slots=True)
class CheckpointStateResolver[StateT: State = State]:
    """Shared lazy state resolver used during one checkpoint restore session.

    The resolver owns the restore-session decode cache. Multiple thin
    checkpoint-backed handles for the same node id intentionally share this
    resolver so the underlying checkpoint anchor/delta chain is decoded only
    once per node.
    """

    state_codec: IncrementalStateCheckpointCodec[StateT]
    state_payloads_by_node_id: Mapping[int, CheckpointNodeStatePayload]
    parent_ids_by_node_id: Mapping[int, int | None]
    _resolved_states: dict[int, StateT] = field(default_factory=dict, init=False)

    def resolve(self, node_id: int) -> StateT:
        """Return one checkpointed state, decoding it only once."""
        resolved_state = self._resolved_states.get(node_id)
        if resolved_state is not None:
            return resolved_state

        state_payload = self.state_payloads_by_node_id[node_id]
        if isinstance(state_payload, AnchorCheckpointStatePayload):
            resolved_state = self._resolve_anchor(state_payload)
        else:
            resolved_state = self._resolve_delta(
                node_id=node_id,
                state_payload=state_payload,
            )
        self._resolved_states[node_id] = resolved_state
        return resolved_state

    def summary(self, node_id: int) -> CheckpointStateSummary | None:
        """Return optional checkpoint summary metadata for ``node_id``."""
        state_payload = self.state_payloads_by_node_id[node_id]
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
        """Resolve one delta payload by first resolving its representative parent."""
        parent_node_id = self.parent_ids_by_node_id[node_id]
        if parent_node_id is None:
            raise KeyError(f"Delta checkpoint node {node_id} has no parent node id.")

        parent_state = self.resolve(parent_node_id)
        return self.state_codec.load_delta_from_parent(
            parent_state=parent_state,
            delta_ref=state_payload.delta_ref,
        )


@dataclass(frozen=True, slots=True)
class CheckpointBackedStateHandle[StateT: State = State]:
    """Thin lazy handle that resolves one checkpointed node on first access."""

    resolver: CheckpointStateResolver[StateT]
    node_id: int

    def get(self) -> StateT:
        """Resolve and return the concrete state for this checkpoint node."""
        return self.resolver.resolve(self.node_id)
