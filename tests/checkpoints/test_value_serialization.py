"""Focused tests for recursive checkpoint atom serialization."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import pytest
from valanga import Outcome, OverEvent
from valanga.evaluations import Certainty, Value

from anemone.checkpoints import (
    CheckpointSerializationError,
    deserialize_checkpoint_atom,
    deserialize_value,
    serialize_checkpoint_atom,
    serialize_value,
)
from anemone.node_evaluation.common import canonical_value


class _DummyTermination(Enum):
    """Small Enum used to verify tagged enum checkpoint atoms roundtrip."""

    NO_LEGAL_MOVES = "no-legal-moves"


def test_simple_tuple_atom_roundtrip() -> None:
    """Tuple atoms should roundtrip with the explicit tagged payload shape."""
    value = (1, 2, 3, 4)

    payload = serialize_checkpoint_atom(value)

    assert payload == {"type": "tuple", "items": [1, 2, 3, 4]}
    assert deserialize_checkpoint_atom(payload) == value


def test_nested_tuple_atoms_roundtrip() -> None:
    """Nested tuple atoms should deserialize recursively without flattening."""
    value = (1, (2, 3), (4, (5, 6)))

    assert deserialize_checkpoint_atom(serialize_checkpoint_atom(value)) == value


def test_enum_atom_roundtrip_uses_tagged_payload() -> None:
    """Enum atoms should use the canonical tagged dict checkpoint format."""
    payload = serialize_checkpoint_atom(_DummyTermination.NO_LEGAL_MOVES)

    assert payload == {
        "type": "enum",
        "module": __name__,
        "qualname": "_DummyTermination",
        "name": "NO_LEGAL_MOVES",
    }
    assert deserialize_checkpoint_atom(payload) is _DummyTermination.NO_LEGAL_MOVES


def test_mixed_tuple_atom_types_roundtrip() -> None:
    """Tuple atoms should preserve mixed scalar atom members exactly."""
    value = (1, "a", 3.5, True, None)

    assert deserialize_checkpoint_atom(serialize_checkpoint_atom(value)) == value


def test_value_line_roundtrip_preserves_tuple_branch_labels() -> None:
    """Value.line should keep list semantics while preserving tuple branch atoms."""
    value = replace(
        canonical_value.make_estimate_value(score=0.25),
        line=[(1, 2, 3, 4), (5, 6, 7, 8)],
    )

    assert deserialize_value(serialize_value(value)) == value


def test_terminal_value_over_event_roundtrip_preserves_enum_atoms() -> None:
    """Values with over_event enum atoms should roundtrip through checkpoints."""
    value = Value(
        score=1.0,
        certainty=Certainty.TERMINAL,
        over_event=OverEvent(
            outcome=Outcome.WIN,
            termination=_DummyTermination.NO_LEGAL_MOVES,
        ),
        line=None,
    )

    assert deserialize_value(serialize_value(value)) == value


def test_malformed_tuple_payload_raises() -> None:
    """Tuple payloads without a list-valued items field should fail loudly."""
    payload = {"type": "tuple", "items": "not-a-list"}

    with pytest.raises(CheckpointSerializationError):
        deserialize_checkpoint_atom(payload)


def test_malformed_enum_payload_raises() -> None:
    """Enum payloads without the canonical tagged fields should fail clearly."""
    payload = {"type": "enum", "module": __name__, "qualname": "_DummyTermination"}

    with pytest.raises(CheckpointSerializationError):
        deserialize_checkpoint_atom(payload)


def test_value_line_list_behavior_is_unchanged() -> None:
    """Value.line should still roundtrip as a list of atoms."""
    value = replace(canonical_value.make_estimate_value(score=0.5), line=[1, 2, 3])

    restored = deserialize_value(serialize_value(value))

    assert isinstance(restored.line, list)
    assert restored.line == [1, 2, 3]
