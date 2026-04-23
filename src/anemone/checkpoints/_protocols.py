"""Thin runtime-checkable mirrors of the Valanga checkpoint protocols.

This module exists so Anemone can use ``isinstance(..., Protocol)`` checks even
when the local runtime does not yet expose the new Valanga checkpoint module.
The method names and semantics must therefore mirror Valanga exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from valanga import BranchKey, State

if TYPE_CHECKING:
    from valanga.checkpoints import CheckpointStateSummary
else:
    CheckpointStateSummary = object


@runtime_checkable
class IncrementalStateCheckpointCodec[StateT: State = State](Protocol):
    """Runtime-checkable adapter mirroring Valanga's incremental codec API."""

    def dump_anchor_ref(self, state: StateT) -> object:
        """Serialize one anchor snapshot reference for ``state``."""
        ...

    def dump_delta_from_parent(
        self,
        *,
        parent_state: StateT,
        child_state: StateT,
        branch_from_parent: BranchKey | None = None,
    ) -> object:
        """Serialize one child delta relative to a concrete parent state."""
        ...

    def load_anchor_ref(self, anchor_ref: object) -> StateT:
        """Restore one concrete state from an anchor snapshot reference."""
        ...

    def load_child_from_delta(
        self,
        *,
        parent_state: StateT,
        delta_ref: object,
        branch_from_parent: BranchKey | None = None,
    ) -> StateT:
        """Restore one child state by applying ``delta_ref`` to ``parent_state``."""
        ...


@runtime_checkable
class StateCheckpointSummaryCodec[StateT: State = State](Protocol):
    """Optional Valanga summary surface used by checkpoint exporters."""

    def dump_state_summary(self, state: StateT) -> CheckpointStateSummary:
        """Serialize lightweight checkpoint metadata for ``state``."""
        ...


__all__ = [
    "CheckpointStateSummary",
    "IncrementalStateCheckpointCodec",
    "StateCheckpointSummaryCodec",
]
