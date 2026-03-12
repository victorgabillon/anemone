"""Breakpoint models and matchers for live debug sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from .events import (
    BackupFinished,
    BackupStarted,
    DirectValueAssigned,
    NodeOpeningPlanned,
    NodeSelected,
    SearchIterationStarted,
)

if TYPE_CHECKING:
    from .events import SearchDebugEvent

type BackupFlagName = Literal["value_changed", "pv_changed", "over_changed"]


@dataclass(frozen=True)
class EventTypeBreakpoint:
    """Pause when the event class name matches ``event_type``."""

    id: str
    enabled: bool
    event_type: str


@dataclass(frozen=True)
class NodeIdBreakpoint:
    """Pause when an event with a single ``node_id`` matches ``node_id``."""

    id: str
    enabled: bool
    node_id: str


@dataclass(frozen=True)
class BackupFlagBreakpoint:
    """Pause when a ``BackupFinished`` event sets one backup flag to ``True``."""

    id: str
    enabled: bool
    flag_name: BackupFlagName


@dataclass(frozen=True)
class IterationBreakpoint:
    """Pause when an iteration-start event reaches ``iteration_index``."""

    id: str
    enabled: bool
    iteration_index: int


type DebugBreakpoint = (
    EventTypeBreakpoint | NodeIdBreakpoint | BackupFlagBreakpoint | IterationBreakpoint
)


class InvalidBreakpointPayloadError(ValueError):
    """Breakpoint JSON payload contains an invalid value."""

    @classmethod
    def missing_required_fields(cls) -> InvalidBreakpointPayloadError:
        """Build the missing-required-fields error."""
        return cls("Breakpoint JSON is missing required fields")

    @classmethod
    def invalid_backup_flag(cls) -> InvalidBreakpointPayloadError:
        """Build the invalid-backup-flag error."""
        return cls("Backup flag breakpoint requires a valid flag_name")

    @classmethod
    def unsupported_kind(cls, kind: object) -> InvalidBreakpointPayloadError:
        """Build the unsupported-kind error."""
        return cls(f"Unsupported breakpoint kind: {kind!r}")


class InvalidBreakpointPayloadTypeError(TypeError):
    """Breakpoint JSON payload contains an invalid type."""

    @classmethod
    def invalid_enabled_state(cls) -> InvalidBreakpointPayloadTypeError:
        """Build the invalid-enabled-state error."""
        return cls("Breakpoint JSON has invalid enabled state")

    @classmethod
    def invalid_event_type(cls) -> InvalidBreakpointPayloadTypeError:
        """Build the invalid-event-type error."""
        return cls("Event type breakpoint requires an event_type string")

    @classmethod
    def invalid_node_id(cls) -> InvalidBreakpointPayloadTypeError:
        """Build the invalid-node-id error."""
        return cls("Node id breakpoint requires a node_id string")

    @classmethod
    def invalid_iteration_index(cls) -> InvalidBreakpointPayloadTypeError:
        """Build the invalid-iteration-index error."""
        return cls("Iteration breakpoint requires an integer iteration_index")

    @classmethod
    def invalid_breakpoint_entry(cls) -> InvalidBreakpointPayloadTypeError:
        """Build the invalid-breakpoint-entry error."""
        return cls("Breakpoint payload entries must be objects")

    @classmethod
    def unsupported_breakpoint_type(
        cls, breakpoint_spec: object
    ) -> InvalidBreakpointPayloadTypeError:
        """Build the unsupported-breakpoint-type error."""
        return cls(f"Unsupported breakpoint type: {type(breakpoint_spec)!r}")


def breakpoint_matches(
    breakpoint_spec: object,
    event: SearchDebugEvent,
) -> bool:
    """Return whether ``breakpoint_spec`` matches ``event``."""
    if getattr(breakpoint_spec, "enabled", False) is not True:
        return False

    if isinstance(breakpoint_spec, EventTypeBreakpoint):
        return event.__class__.__name__ == breakpoint_spec.event_type
    if isinstance(breakpoint_spec, NodeIdBreakpoint):
        event_node_id = _event_node_id(event)
        return event_node_id == breakpoint_spec.node_id
    if isinstance(breakpoint_spec, BackupFlagBreakpoint):
        if not isinstance(event, BackupFinished):
            return False
        return bool(getattr(event, breakpoint_spec.flag_name))
    if isinstance(breakpoint_spec, IterationBreakpoint):
        return (
            isinstance(event, SearchIterationStarted)
            and event.iteration_index == breakpoint_spec.iteration_index
        )
    return False


def any_breakpoint_matches(
    breakpoint_specs: tuple[DebugBreakpoint, ...],
    event: SearchDebugEvent,
) -> DebugBreakpoint | None:
    """Return the first enabled breakpoint that matches ``event``."""
    for breakpoint_spec in breakpoint_specs:
        if breakpoint_matches(breakpoint_spec, event):
            return breakpoint_spec
    return None


def breakpoint_to_json(breakpoint_spec: object) -> dict[str, object]:
    """Serialize one breakpoint to a JSON-friendly dictionary."""
    if isinstance(breakpoint_spec, EventTypeBreakpoint):
        return {
            "kind": "event_type",
            "id": breakpoint_spec.id,
            "enabled": breakpoint_spec.enabled,
            "event_type": breakpoint_spec.event_type,
        }
    if isinstance(breakpoint_spec, NodeIdBreakpoint):
        return {
            "kind": "node_id",
            "id": breakpoint_spec.id,
            "enabled": breakpoint_spec.enabled,
            "node_id": breakpoint_spec.node_id,
        }
    if isinstance(breakpoint_spec, BackupFlagBreakpoint):
        return {
            "kind": "backup_flag",
            "id": breakpoint_spec.id,
            "enabled": breakpoint_spec.enabled,
            "flag_name": breakpoint_spec.flag_name,
        }
    if isinstance(breakpoint_spec, IterationBreakpoint):
        return {
            "kind": "iteration",
            "id": breakpoint_spec.id,
            "enabled": breakpoint_spec.enabled,
            "iteration_index": breakpoint_spec.iteration_index,
        }
    raise InvalidBreakpointPayloadTypeError.unsupported_breakpoint_type(breakpoint_spec)


def breakpoint_from_json(data: dict[str, object]) -> DebugBreakpoint:
    """Deserialize one breakpoint from a JSON-friendly dictionary."""
    kind = data.get("kind")
    breakpoint_id = data.get("id")
    enabled = data.get("enabled")

    if not isinstance(kind, str) or not isinstance(breakpoint_id, str):
        raise InvalidBreakpointPayloadError.missing_required_fields()
    if not isinstance(enabled, bool):
        raise InvalidBreakpointPayloadTypeError.invalid_enabled_state()

    if kind == "event_type":
        event_type = data.get("event_type")
        if not isinstance(event_type, str):
            raise InvalidBreakpointPayloadTypeError.invalid_event_type()
        return EventTypeBreakpoint(
            id=breakpoint_id,
            enabled=enabled,
            event_type=event_type,
        )

    if kind == "node_id":
        node_id = data.get("node_id")
        if not isinstance(node_id, str):
            raise InvalidBreakpointPayloadTypeError.invalid_node_id()
        return NodeIdBreakpoint(
            id=breakpoint_id,
            enabled=enabled,
            node_id=node_id,
        )

    if kind == "backup_flag":
        flag_name = data.get("flag_name")
        if flag_name not in {"value_changed", "pv_changed", "over_changed"}:
            raise InvalidBreakpointPayloadError.invalid_backup_flag()
        return BackupFlagBreakpoint(
            id=breakpoint_id,
            enabled=enabled,
            flag_name=cast("BackupFlagName", flag_name),
        )

    if kind == "iteration":
        iteration_index = data.get("iteration_index")
        if not isinstance(iteration_index, int):
            raise InvalidBreakpointPayloadTypeError.invalid_iteration_index()
        return IterationBreakpoint(
            id=breakpoint_id,
            enabled=enabled,
            iteration_index=iteration_index,
        )

    raise InvalidBreakpointPayloadError.unsupported_kind(kind)


def breakpoints_to_json(
    breakpoint_specs: tuple[DebugBreakpoint, ...],
) -> list[dict[str, object]]:
    """Serialize breakpoints to a JSON-friendly list."""
    return [breakpoint_to_json(breakpoint_spec) for breakpoint_spec in breakpoint_specs]


def breakpoints_from_json(data: list[object]) -> tuple[DebugBreakpoint, ...]:
    """Deserialize breakpoints from a JSON-friendly list."""
    deserialized: list[DebugBreakpoint] = []
    for item in data:
        if not isinstance(item, dict):
            raise InvalidBreakpointPayloadTypeError.invalid_breakpoint_entry()
        deserialized.append(breakpoint_from_json(cast("dict[str, object]", item)))
    return tuple(deserialized)


def _event_node_id(event: SearchDebugEvent) -> str | None:
    """Return the single-node identifier carried by supported event types."""
    match event:
        case NodeSelected(node_id=node_id):
            return node_id
        case NodeOpeningPlanned(node_id=node_id):
            return node_id
        case DirectValueAssigned(node_id=node_id):
            return node_id
        case BackupStarted(node_id=node_id):
            return node_id
        case BackupFinished(node_id=node_id):
            return node_id
        case _:
            return None


__all__ = [
    "BackupFlagBreakpoint",
    "BackupFlagName",
    "DebugBreakpoint",
    "EventTypeBreakpoint",
    "InvalidBreakpointPayloadError",
    "InvalidBreakpointPayloadTypeError",
    "IterationBreakpoint",
    "NodeIdBreakpoint",
    "any_breakpoint_matches",
    "breakpoint_from_json",
    "breakpoint_matches",
    "breakpoint_to_json",
    "breakpoints_from_json",
    "breakpoints_to_json",
]
