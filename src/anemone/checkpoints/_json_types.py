"""Small type helpers for decoded checkpoint JSON payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping


class CheckpointJsonTypeError(ValueError):
    """Raised when decoded checkpoint JSON has an unexpected shape."""


def _type_error(
    field_name: str, expected: str, value: object
) -> CheckpointJsonTypeError:
    return CheckpointJsonTypeError(
        f"checkpoint JSON field {field_name!r} must be {expected}; "
        f"got {type(value).__name__}"
    )


def require_mapping(value: object, *, field_name: str) -> dict[str, object]:
    """Return ``value`` as a JSON object mapping."""
    if not isinstance(value, dict):
        raise _type_error(field_name, "mapping", value)
    return cast("dict[str, object]", value)


def optional_mapping(value: object, *, field_name: str) -> dict[str, object] | None:
    """Return an optional JSON object mapping."""
    if value is None:
        return None
    return require_mapping(value, field_name=field_name)


def require_mapping_field(
    mapping: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    """Return a required mapping field from ``mapping``."""
    return require_mapping(mapping.get(key), field_name=key)


def optional_mapping_field(
    mapping: Mapping[str, object],
    key: str,
) -> dict[str, object] | None:
    """Return an optional mapping field from ``mapping``."""
    return optional_mapping(mapping.get(key), field_name=key)


def require_list(value: object, *, field_name: str) -> list[object]:
    """Return ``value`` as a JSON list."""
    if not isinstance(value, list):
        raise _type_error(field_name, "list", value)
    return cast("list[object]", value)


def optional_list(value: object, *, field_name: str) -> list[object]:
    """Return an optional JSON list, defaulting missing values to an empty list."""
    if value is None:
        return []
    return require_list(value, field_name=field_name)


def require_list_field(mapping: Mapping[str, object], key: str) -> list[object]:
    """Return a required list field from ``mapping``."""
    return require_list(mapping.get(key), field_name=key)


def optional_list_field(mapping: Mapping[str, object], key: str) -> list[object]:
    """Return an optional list field from ``mapping``."""
    return optional_list(mapping.get(key), field_name=key)


def require_int(value: object, *, field_name: str) -> int:
    """Return ``value`` as an int, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(field_name, "int", value)
    return value


def optional_int(value: object, *, field_name: str) -> int | None:
    """Return an optional int, rejecting booleans."""
    if value is None:
        return None
    return require_int(value, field_name=field_name)


def require_int_field(mapping: Mapping[str, object], key: str) -> int:
    """Return a required int field from ``mapping``."""
    return require_int(mapping.get(key), field_name=key)


def optional_int_field(mapping: Mapping[str, object], key: str) -> int | None:
    """Return an optional int field from ``mapping``."""
    return optional_int(mapping.get(key), field_name=key)


def require_bool(value: object, *, field_name: str) -> bool:
    """Return ``value`` as a bool."""
    if not isinstance(value, bool):
        raise _type_error(field_name, "bool", value)
    return value


def require_bool_field(mapping: Mapping[str, object], key: str) -> bool:
    """Return a required bool field from ``mapping``."""
    return require_bool(mapping.get(key), field_name=key)


def require_float(value: object, *, field_name: str) -> float:
    """Return ``value`` as a float, accepting JSON integer numbers."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _type_error(field_name, "float", value)
    return float(value)


def require_float_field(mapping: Mapping[str, object], key: str) -> float:
    """Return a required float field from ``mapping``."""
    return require_float(mapping.get(key), field_name=key)


def require_str(value: object, *, field_name: str) -> str:
    """Return ``value`` as a string."""
    if not isinstance(value, str):
        raise _type_error(field_name, "str", value)
    return value


def optional_str(value: object, *, field_name: str) -> str | None:
    """Return an optional string."""
    if value is None:
        return None
    return require_str(value, field_name=field_name)


def require_str_field(mapping: Mapping[str, object], key: str) -> str:
    """Return a required string field from ``mapping``."""
    return require_str(mapping.get(key), field_name=key)


def optional_str_field(mapping: Mapping[str, object], key: str) -> str | None:
    """Return an optional string field from ``mapping``."""
    return optional_str(mapping.get(key), field_name=key)
