"""Live debug control helpers for commands, breakpoints, and auto-pause.

The live debug controller is intentionally layered on top of an existing
exploration object rather than modifying core search code in place:

``tree_exploration``
    base runtime object owned by the search implementation
``ControlledTreeExploration``
    shallow cloned facade that swaps in debug wrappers
``_ControlledStoppingCriterion``
    pauses exploration cleanly at iteration boundaries
``_ControlledTreeManager``
    finalizes one-step mode after index updates complete
``_ControlledNodeSelector``
    applies best-effort forced node expansion requests when selector and
    dynamics collaborators are both available

These wrappers must stay transparent when their optional collaborators are
missing. The debug layer may request future behavior, but it must not become a
second search engine or crash the wrapped exploration because one optional
component is absent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
)

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
    EXPAND_NODE = "expand_node"
    RUN_UNTIL_NODE_EVENT = "run_until_node_event"
    RUN_UNTIL_NODE_VALUE_CHANGE = "run_until_node_value_change"
    FOCUS_NODE_TIMELINE = "focus_node_timeline"
    CLEAR_TIMELINE_FOCUS = "clear_timeline_focus"


@dataclass(frozen=True, slots=True)
class _ControllerBreakpointHit:
    """Synthetic hit object for transient controller-driven pause conditions."""

    id: str


def write_debug_command(
    directory: str | Path,
    command: DebugCommand,
    *,
    timestamp: float | None = None,
    extra_payload: dict[str, object] | None = None,
) -> Path:
    """Write one command into ``directory``/``commands.json``."""
    target_directory = Path(directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    target_path = target_directory / "commands.json"
    command_payload: dict[str, object] = {
        "command": command.value,
        "timestamp": time.time() if timestamp is None else timestamp,
    }
    if extra_payload is not None:
        command_payload.update(extra_payload)
    target_path.write_text(
        json.dumps(command_payload, indent=2) + "\n",
        encoding="utf-8",
    )
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
        self._requested_expansion_node_id: str | None = None
        self._run_until_node_event_node_id: str | None = None
        self._run_until_node_value_change_node_id: str | None = None
        self._focused_node_id: str | None = None

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

        self._apply_command(command, payload)
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

    def handle_event(
        self, event: SearchDebugEvent
    ) -> DebugBreakpoint | _ControllerBreakpointHit | None:
        """Auto-pause when the first matching enabled breakpoint is observed."""
        self.poll()

        transient_hit = self._matching_transient_hit(event)
        if transient_hit is not None:
            self._paused = True
            self._step_requested = False
            self._last_breakpoint_hit = transient_hit.id
            self.write_control_state()
            return transient_hit

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

    def expand_node(self, node_id: str) -> None:
        """Queue ``node_id`` for best-effort expansion on a future iteration."""
        write_debug_command(
            self._directory,
            DebugCommand.EXPAND_NODE,
            extra_payload={"node_id": node_id},
        )
        self._requested_expansion_node_id = node_id
        self.write_control_state()

    def run_until_node_event(self, node_id: str) -> None:
        """Resume exploration until an event involving ``node_id`` is observed."""
        write_debug_command(
            self._directory,
            DebugCommand.RUN_UNTIL_NODE_EVENT,
            extra_payload={"node_id": node_id},
        )
        self._paused = False
        self._step_requested = False
        self._run_until_node_event_node_id = node_id
        self.write_control_state()

    def run_until_node_value_change(self, node_id: str) -> None:
        """Resume exploration until ``node_id`` reports a value-changing backup."""
        write_debug_command(
            self._directory,
            DebugCommand.RUN_UNTIL_NODE_VALUE_CHANGE,
            extra_payload={"node_id": node_id},
        )
        self._paused = False
        self._step_requested = False
        self._run_until_node_value_change_node_id = node_id
        self.write_control_state()

    def focus_node_timeline(self, node_id: str) -> None:
        """Persist that the live browser timeline should focus on ``node_id``."""
        write_debug_command(
            self._directory,
            DebugCommand.FOCUS_NODE_TIMELINE,
            extra_payload={"node_id": node_id},
        )
        self._focused_node_id = node_id
        self.write_control_state()

    def clear_timeline_focus(self) -> None:
        """Clear any live browser timeline node focus."""
        write_debug_command(self._directory, DebugCommand.CLEAR_TIMELINE_FOCUS)
        self._focused_node_id = None
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

    def requested_expansion_node_id(self) -> str | None:
        """Return the queued best-effort expansion target after polling commands."""
        self.poll()
        return self._requested_expansion_node_id

    def consume_requested_expansion(self, node_id: str) -> None:
        """Clear the queued expansion request when ``node_id`` has been handled."""
        if self._requested_expansion_node_id != node_id:
            return

        self._requested_expansion_node_id = None
        self.write_control_state()

    def write_control_state(self) -> Path:
        """Persist the current control/breakpoint state for the browser."""
        payload = {
            "paused": self._paused,
            "step_requested": self._step_requested,
            "last_breakpoint_hit": self._last_breakpoint_hit,
            "requested_expansion_node_id": self._requested_expansion_node_id,
            "run_until_node_event_node_id": self._run_until_node_event_node_id,
            "run_until_node_value_change_node_id": (
                self._run_until_node_value_change_node_id
            ),
            "focused_node_id": self._focused_node_id,
            "breakpoints": breakpoints_to_json(self.load_breakpoints()),
        }
        self._control_state_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return self._control_state_path

    def _apply_command(self, command: DebugCommand, payload: dict[str, object]) -> None:
        """Update in-memory control state from one parsed command."""
        node_id = payload.get("node_id")
        valid_node_id = node_id if isinstance(node_id, str) and node_id else None
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
            case DebugCommand.EXPAND_NODE:
                self._requested_expansion_node_id = valid_node_id
            case DebugCommand.RUN_UNTIL_NODE_EVENT:
                self._paused = False
                self._step_requested = False
                self._run_until_node_event_node_id = valid_node_id
            case DebugCommand.RUN_UNTIL_NODE_VALUE_CHANGE:
                self._paused = False
                self._step_requested = False
                self._run_until_node_value_change_node_id = valid_node_id
            case DebugCommand.FOCUS_NODE_TIMELINE:
                self._focused_node_id = valid_node_id
            case DebugCommand.CLEAR_TIMELINE_FOCUS:
                self._focused_node_id = None
            case DebugCommand.NONE:
                return

    def _load_control_state(self) -> None:
        """Load any previously persisted control state from disk."""
        payload = self._read_json_dict(self._control_state_path)
        paused = payload.get("paused")
        step_requested = payload.get("step_requested")
        last_breakpoint_hit = payload.get("last_breakpoint_hit")
        requested_expansion_node_id = payload.get("requested_expansion_node_id")
        run_until_node_event_node_id = payload.get("run_until_node_event_node_id")
        run_until_node_value_change_node_id = payload.get(
            "run_until_node_value_change_node_id"
        )
        focused_node_id = payload.get("focused_node_id")

        if isinstance(paused, bool):
            self._paused = paused
        if isinstance(step_requested, bool):
            self._step_requested = step_requested
        if isinstance(last_breakpoint_hit, str) or last_breakpoint_hit is None:
            self._last_breakpoint_hit = last_breakpoint_hit
        if (
            isinstance(requested_expansion_node_id, str)
            or requested_expansion_node_id is None
        ):
            self._requested_expansion_node_id = requested_expansion_node_id
        if (
            isinstance(run_until_node_event_node_id, str)
            or run_until_node_event_node_id is None
        ):
            self._run_until_node_event_node_id = run_until_node_event_node_id
        if (
            isinstance(run_until_node_value_change_node_id, str)
            or run_until_node_value_change_node_id is None
        ):
            self._run_until_node_value_change_node_id = (
                run_until_node_value_change_node_id
            )
        if isinstance(focused_node_id, str) or focused_node_id is None:
            self._focused_node_id = focused_node_id

    def _matching_transient_hit(
        self, event: SearchDebugEvent
    ) -> _ControllerBreakpointHit | None:
        """Return a synthetic hit when one transient run-until condition matches."""
        if self._run_until_node_event_node_id is not None and _event_involves_node_id(
            event, self._run_until_node_event_node_id
        ):
            node_id = self._run_until_node_event_node_id
            self._run_until_node_event_node_id = None
            return _ControllerBreakpointHit(id=f"node-event:{node_id}")

        if (
            self._run_until_node_value_change_node_id is not None
            and _event_is_value_change_for_node(
                event,
                self._run_until_node_value_change_node_id,
            )
        ):
            node_id = self._run_until_node_value_change_node_id
            self._run_until_node_value_change_node_id = None
            return _ControllerBreakpointHit(id=f"node-value-change:{node_id}")

        return None

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
        if self._base is None:
            raise AttributeError(name)
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
        if self._base is None:
            raise AttributeError(name)
        return getattr(self._base, name)

    def refresh_exploration_indices(self, *, tree: Any) -> None:
        """Delegate index refresh and consume a pending single-step."""
        refresh = getattr(self._base, "refresh_exploration_indices", None)
        if refresh is not None:
            refresh(tree=tree)
        else:
            self._base.update_indices(tree=tree)
        self._controller.complete_iteration()

    def update_indices(self, *, tree: Any) -> None:
        """Backward-compatible alias for ``refresh_exploration_indices``."""
        self.refresh_exploration_indices(tree=tree)


class _ControlledNodeSelector:
    """Wrap a node selector and optionally force one node expansion.

    If no expansion is queued, behavior should remain identical to the wrapped
    selector. The wrapper must also tolerate missing optional collaborators and
    degrade to a harmless no-op rather than crashing exploration.
    """

    def __init__(
        self,
        base: Any,
        *,
        controller: DebugSessionController,
        dynamics: Any,
    ) -> None:
        self._base = base
        self._controller = controller
        self._dynamics = dynamics

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped node selector."""
        if self._base is None:
            raise AttributeError(name)
        return getattr(self._base, name)

    def choose_node_and_branch_to_open(
        self,
        *,
        tree: Any,
        latest_tree_expansions: Any,
    ) -> Any:
        """Prefer a queued node-expansion request when it can be fulfilled."""
        if self._base is None:
            return OpeningInstructions()

        requested_node_id = self._controller.requested_expansion_node_id()
        if requested_node_id is not None:
            forced_instructions = _build_forced_opening_instructions(
                tree=tree,
                node_id=requested_node_id,
                dynamics=self._dynamics,
            )
            if forced_instructions is not None:
                self._controller.consume_requested_expansion(requested_node_id)
                return forced_instructions

        return self._base.choose_node_and_branch_to_open(
            tree=tree,
            latest_tree_expansions=latest_tree_expansions,
        )


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
        """Build a controlled exploration facade from an existing exploration.

        Wrappers are only applied when the corresponding base collaborator is
        present. Optional collaborators like ``node_selector`` and ``dynamics``
        are left untouched when absent so the controlled facade remains
        transparent.
        """
        tree_manager = getattr(tree_exploration, "tree_manager", None)
        updates: dict[str, Any] = {
            "tree_manager": _ControlledTreeManager(
                tree_manager,
                controller=controller,
            ),
            "stopping_criterion": _ControlledStoppingCriterion(
                tree_exploration.stopping_criterion,
                controller=controller,
            ),
        }
        node_selector = getattr(tree_exploration, "node_selector", None)
        dynamics = (
            getattr(tree_manager, "dynamics", None)
            if tree_manager is not None
            else None
        )
        if node_selector is not None and dynamics is not None:
            updates["node_selector"] = _ControlledNodeSelector(
                node_selector,
                controller=controller,
                dynamics=dynamics,
            )

        controlled_exploration = clone_exploration(tree_exploration, **updates)

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


def _event_involves_node_id(event: Any, node_id: str) -> bool:
    """Return whether ``event`` references ``node_id`` in its structured ids."""
    for attribute_name in ("node_id", "parent_id", "child_id"):
        if getattr(event, attribute_name, None) == node_id:
            return True
    return False


def _event_is_value_change_for_node(event: Any, node_id: str) -> bool:
    """Return whether ``event`` is a value-changing backup for ``node_id``."""
    return (
        getattr(event, "node_id", None) == node_id
        and getattr(event, "value_changed", None) is True
    )


def _is_any_list(value: object) -> TypeGuard[list[Any]]:
    """Return whether ``value`` is a list with element type intentionally erased."""
    return isinstance(value, list)


def _find_tree_node_by_id(root: Any, node_id: str) -> Any | None:
    """Return the first reachable node whose ``id`` matches ``node_id``."""
    seen_node_ids: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue

        node_identity = id(node)
        if node_identity in seen_node_ids:
            continue
        seen_node_ids.add(node_identity)

        if str(getattr(node, "id", "")) == node_id:
            return node

        branches_children = cast(
            "dict[Any, Any | None]",
            getattr(node, "branches_children", None) or {},
        )
        stack.extend(child for child in branches_children.values() if child is not None)

    return None


def _build_forced_opening_instructions(tree: Any, node_id: str, dynamics: Any) -> Any:
    """Return best-effort opening instructions for unopened branches on ``node_id``."""
    root_node = getattr(tree, "root_node", None)
    if root_node is None or dynamics is None:
        return None

    node_to_open = _find_tree_node_by_id(root_node, node_id)
    if node_to_open is None:
        return None
    if not hasattr(node_to_open, "id") or not hasattr(node_to_open, "state"):
        return None

    branches_children_value = getattr(node_to_open, "branches_children", None)
    if not isinstance(branches_children_value, dict):
        return None

    legal_actions_provider = getattr(dynamics, "legal_actions", None)
    if not callable(legal_actions_provider):
        return None

    legal_actions_container = legal_actions_provider(node_to_open.state)
    get_all = getattr(legal_actions_container, "get_all", None)
    if not callable(get_all):
        return None

    legal_actions_value = get_all()
    if not _is_any_list(legal_actions_value):
        return None
    legal_actions = legal_actions_value

    branches_children = cast("dict[Any, Any | None]", branches_children_value)
    first_unopened_branch = next(
        (branch for branch in legal_actions if branches_children.get(branch) is None),
        None,
    )
    if first_unopened_branch is None:
        return None

    opening_instructions: OpeningInstructions[Any] = OpeningInstructions()
    opening_instructions[(int(node_to_open.id), first_unopened_branch)] = (
        OpeningInstruction(
            node_to_open=node_to_open,
            branch=first_unopened_branch,
        )
    )
    return opening_instructions


__all__ = [
    "ControlledTreeExploration",
    "DebugCommand",
    "DebugSessionController",
    "write_debug_command",
]
