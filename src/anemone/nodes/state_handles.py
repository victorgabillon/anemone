"""Explicit state-handle abstractions used by core node types."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from valanga import State


class StateHandle[StateT: State = State](Protocol):
    """Minimal explicit handle for retrieving one node state."""

    def get(self) -> StateT:
        """Return the concrete state behind this handle."""
        ...


class _StateRefLoader[StateT: State = State](Protocol):
    """Checkpoint-codec adapter surface needed by checkpoint-backed handles."""

    def load_state_ref(self, state_ref: object) -> StateT:
        """Decode one opaque checkpoint state reference."""
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
    resolver so the underlying checkpoint state reference is decoded only once.
    """

    state_codec: _StateRefLoader[StateT]
    state_refs_by_node_id: Mapping[int, object]
    _resolved_states: dict[int, StateT] = field(default_factory=dict, init=False)

    def resolve(self, node_id: int) -> StateT:
        """Return one checkpointed state, decoding it only once."""
        resolved_state = self._resolved_states.get(node_id)
        if resolved_state is not None:
            return resolved_state

        state_ref = self.state_refs_by_node_id[node_id]
        resolved_state = self.state_codec.load_state_ref(state_ref)
        self._resolved_states[node_id] = resolved_state
        return resolved_state


@dataclass(frozen=True, slots=True)
class CheckpointBackedStateHandle[StateT: State = State]:
    """Thin lazy handle that resolves one checkpointed node on first access."""

    resolver: CheckpointStateResolver[StateT]
    node_id: int

    def get(self) -> StateT:
        """Resolve and return the concrete state for this checkpoint node."""
        return self.resolver.resolve(self.node_id)
