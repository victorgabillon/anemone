"""Public setup helpers for live debug sessions.

This module provides the first stable orchestration layer for the browser-based
debugger. Callers should not need to remember the composition order of:

* ``DebugSessionController``
* ``make_tree_snapshot_provider(...)``
* ``LiveDebugSessionRecorder``
* ``ControlledTreeExploration.from_tree_exploration(...)``

Instead, ``build_live_debug_environment(...)`` assembles those pieces into one
explicit ``LiveDebugEnvironment`` value with a small lifecycle surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .live_control import ControlledTreeExploration, DebugSessionController
from .live_session import LiveDebugSessionRecorder
from .recording import SnapshotPolicy, make_tree_snapshot_provider


@dataclass(slots=True)
class LiveDebugEnvironment:
    """Group the controller, recorder, and wrapped exploration for one session.

    The recorder writes live session files incrementally while exploration runs.
    Calling :meth:`finalize` marks the session complete for the browser viewer.
    Live serving can happen during the run or after it completes. If a process
    exits before finalization, the session remains usable but stays marked as
    incomplete.
    """

    controlled_exploration: Any
    controller: DebugSessionController
    recorder: LiveDebugSessionRecorder
    session_directory: Path

    def finalize(self) -> None:
        """Mark the live session complete for the browser viewer."""
        self.recorder.finalize()


def build_live_debug_environment(
    tree_exploration: Any,
    session_directory: str | Path,
    *,
    snapshot_format: str = "svg",
    snapshot_policy: SnapshotPolicy | None = None,
) -> LiveDebugEnvironment:
    """Assemble the standard live debug components around ``tree_exploration``.

    The returned environment wires, in order:

    1. ``DebugSessionController``
    2. a lazy tree snapshot provider rooted at ``tree_exploration.tree.root_node``
    3. ``LiveDebugSessionRecorder`` using that provider and controller
    4. ``ControlledTreeExploration`` with debug event emission routed to the
       recorder

    The returned environment is the supported public entrypoint for live
    debugging. Call ``controlled_exploration.explore(...)`` to run the search,
    and ``finalize()`` once the run is finished.
    """
    resolved_session_directory = Path(session_directory)
    resolved_session_directory.mkdir(parents=True, exist_ok=True)

    controller = DebugSessionController(resolved_session_directory)
    snapshot_provider = make_tree_snapshot_provider(
        root_getter=lambda: _get_tree_root(tree_exploration),
    )
    recorder = LiveDebugSessionRecorder(
        resolved_session_directory,
        snapshot_provider=snapshot_provider,
        snapshot_policy=snapshot_policy,
        snapshot_format=snapshot_format,
        controller=controller,
    )
    controlled_exploration = ControlledTreeExploration.from_tree_exploration(
        tree_exploration,
        controller=controller,
        debug_sink=recorder,
    )
    return LiveDebugEnvironment(
        controlled_exploration=controlled_exploration,
        controller=controller,
        recorder=recorder,
        session_directory=resolved_session_directory,
    )


def _get_tree_root(tree_exploration: Any) -> Any | None:
    """Return the current root node for ``tree_exploration`` when available."""
    tree = getattr(tree_exploration, "tree", None)
    return getattr(tree, "root_node", None)


__all__ = ["LiveDebugEnvironment", "build_live_debug_environment"]
