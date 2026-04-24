"""Checkpoint-backed lazy state handles used during restore."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from valanga import BranchKey, State

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
    _resolved_states: dict[int, StateT] = field(
        default_factory=lambda: cast("dict[int, StateT]", {}),
        init=False,
    )

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
        """Resolve one delta payload by first resolving its stored state parent."""
        parent_node_id = state_payload.state_parent_node_id
        if parent_node_id == node_id:
            raise CheckpointStateResolutionError.delta_node_missing_parent(node_id)

        parent_state = self.resolve(parent_node_id)
        return self.state_codec.load_child_from_delta(
            parent_state=parent_state,
            delta_ref=state_payload.delta_ref,
            branch_from_parent=cast(
                "BranchKey | None",
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


__all__ = [
    "CheckpointBackedStateHandle",
    "CheckpointStateResolutionError",
    "CheckpointStateResolver",
]
