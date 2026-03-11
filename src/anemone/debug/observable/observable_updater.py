"""Updater wrapper emitting best-effort backup events."""

from __future__ import annotations

import inspect
from typing import Any

from anemone.debug.events import BackupFinished, BackupStarted
from anemone.debug.observable.state_diff import summarize_node_evaluation
from anemone.debug.sink import NullSearchDebugSink, SearchDebugSink


class ObservableUpdater:
    """Wrap an updater and infer backup events from before/after node summaries."""

    def __init__(self, base: Any, debug_sink: SearchDebugSink | None = None) -> None:
        """Store the wrapped updater and the sink receiving inferred events."""
        self._base = base
        self._debug_sink = debug_sink or NullSearchDebugSink()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped updater."""
        return getattr(self._base, name)

    def perform_updates(self, node_to_update: Any, update_instructions: Any) -> Any:
        """Observe backup/update processing for one node."""
        before = summarize_node_evaluation(node_to_update)
        node_id = str(node_to_update.id)
        self._debug_sink.emit(BackupStarted(node_id=node_id))

        result = self._perform_updates(
            node_to_update=node_to_update,
            update_instructions=update_instructions,
        )

        after = summarize_node_evaluation(node_to_update)
        self._debug_sink.emit(
            BackupFinished(
                node_id=node_id,
                value_changed=before.backed_up_value_repr != after.backed_up_value_repr,
                pv_changed=before.pv_repr != after.pv_repr,
                over_changed=before.over_repr != after.over_repr,
            )
        )
        return result

    def _perform_updates(self, *, node_to_update: Any, update_instructions: Any) -> Any:
        """Call the wrapped updater using whichever update-parameter name it exposes."""
        perform_updates = self._base.perform_updates

        try:
            signature = inspect.signature(perform_updates)
        except (TypeError, ValueError):
            return perform_updates(node_to_update, update_instructions)

        parameter_names = signature.parameters
        if "update_instructions" in parameter_names:
            return perform_updates(
                node_to_update=node_to_update,
                update_instructions=update_instructions,
            )
        if "updates_instructions" in parameter_names:
            return perform_updates(
                node_to_update=node_to_update,
                updates_instructions=update_instructions,
            )

        return perform_updates(node_to_update, update_instructions)
