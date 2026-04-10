"""Low-level serialization helpers for checkpointed ``Value`` objects."""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, cast

from valanga import Outcome, OverEvent
from valanga.evaluations import Certainty, Value

from anemone.node_evaluation.common import canonical_value

from .payloads import (
    CheckpointAtomPayload,
    EnumMemberPayload,
    SerializedOverEventPayload,
    SerializedValuePayload,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from anemone._valanga_types import AnyOverEvent


class CheckpointSerializationError(ValueError):
    """Raised when checkpoint serialization cannot represent a runtime object."""

    @classmethod
    def unsupported_atom(cls, value: object) -> CheckpointSerializationError:
        """Return the error for unsupported checkpoint atom values."""
        return cls(
            "Checkpoint payloads currently support only None, bool, int, float, "
            f"str, or Enum members, got {type(value)!r}."
        )

    @classmethod
    def unresolved_enum_type(
        cls,
        *,
        module: str,
        qualname: str,
    ) -> CheckpointSerializationError:
        """Return the error for non-Enum qualname targets."""
        return cls(
            "Checkpoint enum payload did not resolve to an Enum type: "
            f"{module}:{qualname}."
        )

    @classmethod
    def unknown_enum_member(
        cls,
        *,
        member_name: str,
        enum_type: type[Enum],
    ) -> CheckpointSerializationError:
        """Return the error for missing Enum members."""
        return cls(f"Unknown Enum member {member_name!r} for {enum_type!r}.")

    @classmethod
    def invalid_over_event(cls, over_event: object) -> CheckpointSerializationError:
        """Return the error for unsupported over-event payloads."""
        return cls(
            "Expected an OverEvent-like object with a valanga Outcome, "
            f"got {type(over_event)!r}."
        )

    @classmethod
    def unresolved_qualname(
        cls,
        *,
        qualname: str,
        root: object,
    ) -> CheckpointSerializationError:
        """Return the error for qualnames that cannot be resolved."""
        return cls(f"Cannot resolve checkpoint qualname {qualname!r} from {root!r}.")


def serialize_checkpoint_atom(value: object) -> CheckpointAtomPayload:
    """Serialize one small scalar/enum value used inside checkpoint payloads."""
    if isinstance(value, Enum):
        return EnumMemberPayload(
            module=value.__class__.__module__,
            qualname=value.__class__.__qualname__,
            name=value.name,
        )

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    raise CheckpointSerializationError.unsupported_atom(value)


def deserialize_checkpoint_atom(payload: CheckpointAtomPayload) -> object:
    """Deserialize one small scalar/enum checkpoint payload."""
    if not isinstance(payload, EnumMemberPayload):
        return payload

    module = import_module(payload.module)
    enum_type = _resolve_qualname(module, payload.qualname)
    if not isinstance(enum_type, type) or not issubclass(enum_type, Enum):
        raise CheckpointSerializationError.unresolved_enum_type(
            module=payload.module,
            qualname=payload.qualname,
        )

    try:
        return enum_type[payload.name]
    except KeyError as exc:
        raise CheckpointSerializationError.unknown_enum_member(
            member_name=payload.name,
            enum_type=enum_type,
        ) from exc


def serialize_over_event(
    over_event: object | None,
) -> SerializedOverEventPayload | None:
    """Serialize one ``OverEvent``-like object carried by a ``Value``."""
    if over_event is None:
        return None

    outcome = getattr(over_event, "outcome", None)
    if not isinstance(outcome, Outcome):
        raise CheckpointSerializationError.invalid_over_event(over_event)

    return SerializedOverEventPayload(
        outcome=outcome.name,
        termination=serialize_checkpoint_atom(getattr(over_event, "termination", None)),
        winner=serialize_checkpoint_atom(getattr(over_event, "winner", None)),
    )


def deserialize_over_event(
    payload: SerializedOverEventPayload | None,
) -> AnyOverEvent | None:
    """Deserialize one serialized over-event payload."""
    if payload is None:
        return None

    termination = cast("Enum | None", deserialize_checkpoint_atom(payload.termination))
    winner = deserialize_checkpoint_atom(payload.winner)
    return OverEvent(
        outcome=Outcome[payload.outcome],
        termination=termination,
        winner=winner,
    )


def serialize_value(value: Value) -> SerializedValuePayload:
    """Serialize one validated runtime ``Value`` into a checkpoint payload."""
    validated_value = canonical_value.validate_value_semantics(value)
    return SerializedValuePayload(
        score=validated_value.score,
        certainty=validated_value.certainty.name,
        over_event=serialize_over_event(validated_value.over_event),
        line=(
            [serialize_checkpoint_atom(branch) for branch in validated_value.line]
            if validated_value.line is not None
            else None
        ),
    )


def deserialize_value(payload: SerializedValuePayload) -> Value:
    """Deserialize one checkpoint ``Value`` payload."""
    line = cast(
        "list[Hashable] | None",
        (
            [deserialize_checkpoint_atom(branch) for branch in payload.line]
            if payload.line is not None
            else None
        ),
    )
    restored_value = Value(
        score=payload.score,
        certainty=Certainty[payload.certainty],
        over_event=deserialize_over_event(payload.over_event),
        line=line,
    )
    return canonical_value.validate_value_semantics(restored_value)


def _resolve_qualname(root: object, qualname: str) -> object:
    """Resolve a dotted qualname from a module or parent object."""
    current = root
    for part in qualname.split("."):
        try:
            current = getattr(current, part)
        except AttributeError as exc:
            raise CheckpointSerializationError.unresolved_qualname(
                qualname=qualname,
                root=root,
            ) from exc
    return current


__all__ = [
    "CheckpointSerializationError",
    "deserialize_checkpoint_atom",
    "deserialize_over_event",
    "deserialize_value",
    "serialize_checkpoint_atom",
    "serialize_over_event",
    "serialize_value",
]
