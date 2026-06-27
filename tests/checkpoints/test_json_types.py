"""Tests for checkpoint JSON type helpers."""

from __future__ import annotations

import pytest

from anemone.checkpoints._json_types import (
    CheckpointJsonTypeError,
    optional_int,
    optional_list_field,
    require_float,
    require_int,
    require_int_field,
    require_list,
    require_mapping,
)


def test_require_int_accepts_int_and_rejects_bool() -> None:
    """Integer JSON fields should not silently accept booleans."""
    assert require_int(3, field_name="node_id") == 3

    with pytest.raises(CheckpointJsonTypeError, match=r"'node_id'.*int.*bool"):
        require_int(True, field_name="node_id")


@pytest.mark.parametrize("bad_value", ["3", None])
def test_require_int_rejects_non_int_values(bad_value: object) -> None:
    """Integer JSON fields should reject non-integers."""
    with pytest.raises(CheckpointJsonTypeError, match=r"'node_id'.*int"):
        require_int(bad_value, field_name="node_id")


def test_optional_int_accepts_none() -> None:
    """Optional integer JSON fields should allow missing values."""
    assert optional_int(None, field_name="parent_node_id") is None


def test_require_float_accepts_json_numbers_but_rejects_bool() -> None:
    """Float JSON fields should accept integer JSON numbers."""
    assert require_float(2, field_name="score") == 2.0
    assert require_float(2.5, field_name="score") == 2.5

    with pytest.raises(CheckpointJsonTypeError, match=r"'score'.*float.*bool"):
        require_float(False, field_name="score")


def test_require_mapping_and_list_validate_container_shape() -> None:
    """Container helpers should reject the wrong JSON shape."""
    assert require_mapping({"node_id": 1}, field_name="node") == {"node_id": 1}
    assert require_list([{"node_id": 1}], field_name="nodes") == [{"node_id": 1}]

    with pytest.raises(CheckpointJsonTypeError, match=r"'node'.*mapping.*list"):
        require_mapping([], field_name="node")
    with pytest.raises(CheckpointJsonTypeError, match=r"'nodes'.*list.*dict"):
        require_list({}, field_name="nodes")


def test_required_field_helper_reports_missing_field() -> None:
    """Missing required fields should name the expected field."""
    with pytest.raises(CheckpointJsonTypeError, match=r"'node_id'.*int.*NoneType"):
        require_int_field({}, "node_id")


def test_optional_list_field_defaults_missing_value_to_empty_list() -> None:
    """Optional list fields should preserve existing missing-list defaults."""
    assert optional_list_field({}, "linked_children") == []
