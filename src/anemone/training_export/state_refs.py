"""Checkpoint-aware state-ref helpers for training export."""

# pylint: disable=duplicate-code

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from anemone._best_effort import safe_getattr as _safe_getattr
from anemone.checkpoints.state_handles import CheckpointBackedStateHandle

if TYPE_CHECKING:
    from collections.abc import Callable

    from anemone.training_export.builders import TrainingExportProfiler


def state_ref_payload_without_resolving(
    node: object,
    *,
    checkpoint_payload_to_state_ref: Callable[
        [CheckpointBackedStateHandle[Any], object],
        object | None,
    ],
    materialized_state_to_state_ref: Callable[[object], object | None],
    profile: TrainingExportProfiler | None = None,
) -> object | None:
    """Return a training state-ref, preferring checkpoint payloads over state."""
    if profile is not None:
        profile.observe_state_handle(node)

    handle = raw_checkpoint_backed_state_handle(node)
    if handle is not None:
        started_at = perf_counter()
        payload = handle.checkpoint_payload_for_reuse_or_none()
        if payload is not None:
            state_ref_payload = checkpoint_payload_to_state_ref(handle, payload)
            if state_ref_payload is not None:
                if profile is not None:
                    profile.record_state_ref_conversion(perf_counter() - started_at)
                return state_ref_payload

    state_started_at = perf_counter()
    state = _safe_getattr(node, "state")
    state_access_elapsed_s = perf_counter() - state_started_at
    if profile is not None:
        profile.record_state_access(
            state_access_elapsed_s,
            state_present=state is not None,
        )
    if state is None:
        return None

    conversion_started_at = perf_counter()
    state_ref_payload = materialized_state_to_state_ref(state)
    if profile is not None:
        profile.record_state_ref_conversion(perf_counter() - conversion_started_at)
    return state_ref_payload


def raw_checkpoint_backed_state_handle(
    node: object,
) -> CheckpointBackedStateHandle[Any] | None:
    """Return a checkpoint-backed state handle without resolving ``node.state``."""
    handle = _safe_getattr(node, "state_handle")
    if isinstance(handle, CheckpointBackedStateHandle):
        return cast(  # type: ignore[redundant-cast]
            "CheckpointBackedStateHandle[Any]",
            handle,
        )
    tree_node = _safe_getattr(node, "tree_node")
    if tree_node is None:
        return None
    tree_handle = _safe_getattr(tree_node, "state_handle")
    if isinstance(tree_handle, CheckpointBackedStateHandle):
        return cast(  # type: ignore[redundant-cast]
            "CheckpointBackedStateHandle[Any]",
            tree_handle,
        )
    return None


__all__ = [
    "raw_checkpoint_backed_state_handle",
    "state_ref_payload_without_resolving",
]
