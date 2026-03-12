"""Live debug control helpers for commands, breakpoints, and auto-pause."""

from __future__ import annotations

import json
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .breakpoints import (
    DebugBreakpoint,
    any_breakpoint_matches,
    breakpoint_from_json,
    breakpoints_to_json,
)
from .exploration_clone import clone_exploration
from .observable.observable_tree_exploration import ObservableTreeExploration

if TYPE_CHECKING:
    from random import Random

    from .events import SearchDebugEvent
    from .sink import SearchDebugSink


class DebugCommand(StrEnum):
    """Supported live debug commands."""

    NONE = "none"
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"


def write_debug_command(
    directory: str | Path,
    command: DebugCommand,
    *,
    timestamp: float | None = None,
) -> Path:
    """Write one command into ``directory``/``commands.json``."""
    target_directory = Path(directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    target_path = target_directory / "commands.json"
    payload = {
        "command": command.value,
        "timestamp": time.time() if timestamp is None else timestamp,
    }
    target_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target_path


class DebugSessionController:
    """File-backed controller for live debug commands and breakpoints."""

    def __init__(
        self,
        directory: str | Path,
        *,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        """Initialize the controller for one live session directory."""
        self._directory = Path(directory)
        self._commands_path = self._directory / "commands.json"
        self._breakpoints_path = self._directory / "breakpoints.json"
        self._control_state_path = self._directory / "control_state.json"
        self._poll_interval_seconds = poll_interval_seconds
        self._last_command_marker: tuple[str | None, float | int | None] | None = None
        self._paused = False
        self._step_requested = False
        self._last_breakpoint_hit: str | None = None

        self._directory.mkdir(parents=True, exist_ok=True)
        self._load_control_state()
        if not self._commands_path.exists():
            write_debug_command(self._directory, DebugCommand.NONE, timestamp=0)
        if not self._breakpoints_path.exists():
            self._write_breakpoints(())
        self.write_control_state()

    def poll(self) -> DebugCommand:
        """Read and apply the latest command, returning only new commands."""
        if not self._commands_path.exists():
            return DebugCommand.NONE

        payload = self._read_json_dict(self._commands_path)
        command_name = payload.get("command")
        timestamp = payload.get("timestamp")
        marker = (
            command_name if isinstance(command_name, str) else None,
            timestamp if isinstance(timestamp, (float, int)) else None,
        )
        if marker == self._last_command_marker:
            return DebugCommand.NONE

        self._last_command_marker = marker
        if not isinstance(command_name, str):
            return DebugCommand.NONE
        try:
            command = DebugCommand(command_name)
        except ValueError:
            return DebugCommand.NONE

        self._apply_command(command)
        self.write_control_state()
        return command

    def load_breakpoints(self) -> tuple[DebugBreakpoint, ...]:
        """Load persisted breakpoints from disk."""
        payload = self._read_json_list(self._breakpoints_path)
        breakpoints: list[DebugBreakpoint] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                breakpoints.append(
                    breakpoint_from_json(cast("dict[str, object]", item))
                )
            except (TypeError, ValueError):
                continue
        return tuple(breakpoints)

    def save_breakpoints(self, breakpoints: tuple[DebugBreakpoint, ...]) -> None:
        """Persist the provided breakpoints and refresh control state."""
        self._write_breakpoints(breakpoints)
        self.write_control_state()

    def add_breakpoint(self, breakpoint_spec: DebugBreakpoint) -> None:
        """Append one breakpoint to the persisted breakpoint set."""
        self.save_breakpoints((*self.load_breakpoints(), breakpoint_spec))

    def clear_breakpoints(self) -> None:
        """Remove all persisted breakpoints and clear the last hit marker."""
        self._last_breakpoint_hit = None
        self.save_breakpoints(())

    def handle_event(self, event: SearchDebugEvent) -> DebugBreakpoint | None:
        """Auto-pause when the first matching enabled breakpoint is observed."""
        self.poll()
        matching_breakpoint = any_breakpoint_matches(self.load_breakpoints(), event)
        if matching_breakpoint is None:
            return None

        self._paused = True
        self._step_requested = False
        self._last_breakpoint_hit = matching_breakpoint.id
        self.write_control_state()
        return matching_breakpoint

    def request_pause(self) -> None:
        """Request that exploration pause before the next iteration."""
        write_debug_command(self._directory, DebugCommand.PAUSE)
        self._paused = True
        self._step_requested = False
        self.write_control_state()

    def request_resume(self) -> None:
        """Request that exploration resume running normally."""
        write_debug_command(self._directory, DebugCommand.RESUME)
        self._paused = False
        self._step_requested = False
        self.write_control_state()

    def request_step(self) -> None:
        """Request exactly one iteration, then auto-pause again."""
        write_debug_command(self._directory, DebugCommand.STEP)
        self._paused = False
        self._step_requested = True
        self.write_control_state()

    def should_pause(self) -> bool:
        """Return whether exploration should currently remain paused."""
        return self._paused

    def should_step(self) -> bool:
        """Return whether exactly one step is currently pending."""
        return self._step_requested

    def should_pause_before_iteration(self) -> bool:
        """Refresh commands and return whether the next iteration should wait."""
        self.poll()
        return self._paused

    def wait_until_iteration_allowed(self) -> DebugCommand:
        """Block until the controller permits one more iteration."""
        command = self.poll()
        while self.should_pause():
            time.sleep(self._poll_interval_seconds)
            command = self.poll()
        return command

    def consume_step_if_requested(self) -> bool:
        """Consume a pending step and return whether one was active."""
        if not self._step_requested:
            return False

        self._step_requested = False
        self._paused = True
        self.write_control_state()
        return True

    def complete_iteration(self) -> None:
        """Finalize any pending step request after one completed iteration."""
        self.consume_step_if_requested()

    def write_control_state(self) -> Path:
        """Persist the current control/breakpoint state for the browser."""
        payload = {
            "paused": self._paused,
            "step_requested": self._step_requested,
            "last_breakpoint_hit": self._last_breakpoint_hit,
            "breakpoints": breakpoints_to_json(self.load_breakpoints()),
        }
        self._control_state_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return self._control_state_path

    def _apply_command(self, command: DebugCommand) -> None:
        """Update in-memory control state from one parsed command."""
        match command:
            case DebugCommand.PAUSE:
                self._paused = True
                self._step_requested = False
            case DebugCommand.RESUME:
                self._paused = False
                self._step_requested = False
            case DebugCommand.STEP:
                self._paused = False
                self._step_requested = True
            case DebugCommand.NONE:
                return

    def _load_control_state(self) -> None:
        """Load any previously persisted control state from disk."""
        payload = self._read_json_dict(self._control_state_path)
        paused = payload.get("paused")
        step_requested = payload.get("step_requested")
        last_breakpoint_hit = payload.get("last_breakpoint_hit")

        if isinstance(paused, bool):
            self._paused = paused
        if isinstance(step_requested, bool):
            self._step_requested = step_requested
        if isinstance(last_breakpoint_hit, str) or last_breakpoint_hit is None:
            self._last_breakpoint_hit = last_breakpoint_hit

    def _write_breakpoints(self, breakpoints: tuple[DebugBreakpoint, ...]) -> None:
        """Persist breakpoint configuration to ``breakpoints.json``."""
        self._breakpoints_path.write_text(
            json.dumps(breakpoints_to_json(breakpoints), indent=2) + "\n",
            encoding="utf-8",
        )

    def _read_json_dict(self, path: Path) -> dict[str, object]:
        """Read ``path`` as a JSON object or return an empty dictionary."""
        if not path.exists():
            return {}

        try:
            loaded_payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(loaded_payload, dict):
            return {}
        return cast("dict[str, object]", loaded_payload)

    def _read_json_list(self, path: Path) -> list[object]:
        """Read ``path`` as a JSON list or return an empty list."""
        if not path.exists():
            return []

        try:
            loaded_payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(loaded_payload, list):
            return []
        return cast("list[object]", loaded_payload)


class _ControlledStoppingCriterion:
    """Wrap a stopping criterion and block at iteration boundaries."""

    def __init__(self, base: Any, *, controller: DebugSessionController) -> None:
        self._base = base
        self._controller = controller

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped stopping criterion."""
        return getattr(self._base, name)

    def should_we_continue(self, *, tree: Any) -> bool:
        """Delegate continuation checks and wait if the session is paused."""
        should_continue = bool(self._base.should_we_continue(tree=tree))
        if not should_continue:
            return False

        self._controller.wait_until_iteration_allowed()
        return True


class _ControlledTreeManager:
    """Wrap a tree manager and complete pending step mode after one iteration."""

    def __init__(self, base: Any, *, controller: DebugSessionController) -> None:
        self._base = base
        self._controller = controller

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped tree manager."""
        return getattr(self._base, name)

    def update_indices(self, *, tree: Any) -> None:
        """Delegate index updates and consume a pending single-step."""
        self._base.update_indices(tree=tree)
        self._controller.complete_iteration()


class ControlledTreeExploration:
    """Facade over an exploration object with live debug control wrappers."""

    def __init__(
        self, base_exploration: Any, *, controller: DebugSessionController
    ) -> None:
        """Store the controlled exploration facade and its session controller."""
        self._base_exploration = base_exploration
        self._controller = controller

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped exploration."""
        return getattr(self._base_exploration, name)

    @classmethod
    def from_tree_exploration(
        cls,
        tree_exploration: Any,
        *,
        controller: DebugSessionController,
        debug_sink: SearchDebugSink | None = None,
    ) -> ControlledTreeExploration:
        """Build a controlled exploration facade from an existing exploration."""
        controlled_exploration = clone_exploration(
            tree_exploration,
            tree_manager=_ControlledTreeManager(
                tree_exploration.tree_manager,
                controller=controller,
            ),
            stopping_criterion=_ControlledStoppingCriterion(
                tree_exploration.stopping_criterion,
                controller=controller,
            ),
        )

        if debug_sink is not None:
            observed_exploration = ObservableTreeExploration.from_tree_exploration(
                controlled_exploration,
                debug_sink=debug_sink,
            )
            return cls(observed_exploration, controller=controller)

        return cls(controlled_exploration, controller=controller)

    def explore(self, random_generator: Random) -> Any:
        """Delegate exploration to the wrapped controlled exploration object."""
        return self._base_exploration.explore(random_generator=random_generator)


__all__ = [
    "ControlledTreeExploration",
    "DebugCommand",
    "DebugSessionController",
    "write_debug_command",
]
