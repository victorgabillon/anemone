"""Public value-update payloads for live search-tree nodes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def _invalid_node_id_error() -> ValueError:
    """Return the error for invalid public node ids."""
    return ValueError("node_id must be a non-empty string.")


def _invalid_float_type_error(field_name: str) -> TypeError:
    """Return the error for non-numeric scalar fields."""
    return TypeError(f"{field_name} must be a finite float.")


def _non_finite_float_error(field_name: str) -> ValueError:
    """Return the error for non-finite scalar fields."""
    return ValueError(f"{field_name} must be finite.")


def _invalid_optional_bool_error(field_name: str) -> TypeError:
    """Return the error for non-bool optional flag fields."""
    return TypeError(f"{field_name} must be a bool when provided.")


def _single_node_id_collection_error() -> TypeError:
    """Return the error for one string passed where an id collection is expected."""
    return TypeError("node id collections must not be a single string.")


def _node_id_collection_not_iterable_error() -> ValueError:
    """Return the error for non-iterable node id collections."""
    return ValueError("node id collections must be iterable.")


def _negative_result_count_error(field_name: str) -> ValueError:
    """Return the error for negative result counts."""
    return ValueError(f"{field_name} must be non-negative.")


def _applied_count_too_large_error() -> ValueError:
    """Return the error for impossible applied counts."""
    return ValueError("applied_count must not exceed requested_count.")


def _accounted_count_too_large_error() -> ValueError:
    """Return the error for impossible result accounting."""
    return ValueError(
        "applied, missing, and skipped counts must not exceed requested_count."
    )


def _negative_recomputed_count_error() -> ValueError:
    """Return the error for a negative recomputed count."""
    return ValueError("recomputed_count must be non-negative when provided.")


def _validate_node_id(node_id: object) -> str:
    """Return a validated public node id."""
    if not isinstance(node_id, str) or not node_id:
        raise _invalid_node_id_error()
    return node_id


def _validate_finite_float(value: object, *, field_name: str) -> float:
    """Return a validated finite float field."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _invalid_float_type_error(field_name)
    finite_value = float(value)
    if not isfinite(finite_value):
        raise _non_finite_float_error(field_name)
    return finite_value


def _validate_optional_bool(value: object, *, field_name: str) -> bool | None:
    """Return a validated optional bool field."""
    if value is None:
        return None
    if not isinstance(value, bool):
        raise _invalid_optional_bool_error(field_name)
    return value


def _normalize_node_id_tuple(value: tuple[str, ...] | object) -> tuple[str, ...]:
    """Return a validated tuple of public node ids."""
    if isinstance(value, str):
        raise _single_node_id_collection_error()
    try:
        loaded_node_ids: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise _node_id_collection_not_iterable_error() from exc
    return tuple(_validate_node_id(node_id) for node_id in loaded_node_ids)


@dataclass(frozen=True, slots=True)
class NodeValueUpdate:
    """One direct-value update for an existing live tree node."""

    node_id: str
    direct_value: float
    backed_up_value: float | None = None
    is_exact: bool | None = None
    is_terminal: bool | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        """Validate and normalize public update fields."""
        object.__setattr__(self, "node_id", _validate_node_id(self.node_id))
        object.__setattr__(
            self,
            "direct_value",
            _validate_finite_float(self.direct_value, field_name="direct_value"),
        )
        if self.backed_up_value is not None:
            object.__setattr__(
                self,
                "backed_up_value",
                _validate_finite_float(
                    self.backed_up_value,
                    field_name="backed_up_value",
                ),
            )
        object.__setattr__(
            self,
            "is_exact",
            _validate_optional_bool(self.is_exact, field_name="is_exact"),
        )
        object.__setattr__(
            self,
            "is_terminal",
            _validate_optional_bool(self.is_terminal, field_name="is_terminal"),
        )


@dataclass(frozen=True, slots=True)
class NodeValueUpdateResult:
    """Structured summary returned after applying live node-value updates."""

    requested_count: int
    applied_count: int
    missing_node_ids: tuple[str, ...]
    skipped_node_ids: tuple[str, ...] = ()
    recomputed_count: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize public result fields."""
        if self.requested_count < 0:
            raise _negative_result_count_error("requested_count")
        if self.applied_count < 0:
            raise _negative_result_count_error("applied_count")
        if self.applied_count > self.requested_count:
            raise _applied_count_too_large_error()
        object.__setattr__(
            self,
            "missing_node_ids",
            _normalize_node_id_tuple(self.missing_node_ids),
        )
        object.__setattr__(
            self,
            "skipped_node_ids",
            _normalize_node_id_tuple(self.skipped_node_ids),
        )
        accounted_count = (
            self.applied_count + len(self.missing_node_ids) + len(self.skipped_node_ids)
        )
        if accounted_count > self.requested_count:
            raise _accounted_count_too_large_error()
        if self.recomputed_count is not None and self.recomputed_count < 0:
            raise _negative_recomputed_count_error()


__all__ = [
    "NodeValueUpdate",
    "NodeValueUpdateResult",
]
