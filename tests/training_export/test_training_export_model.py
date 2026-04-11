"""Tests for training-export builders and model behavior."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_PACKAGE_ROOT = _REPO_ROOT / "src" / "anemone"

if "anemone" not in sys.modules:
    _stub_package = ModuleType("anemone")
    _stub_package.__path__ = [str(_SRC_PACKAGE_ROOT)]
    sys.modules["anemone"] = _stub_package

from anemone.training_export import (
    TRAINING_TREE_SNAPSHOT_FORMAT_KIND,
    TRAINING_TREE_SNAPSHOT_FORMAT_VERSION,
    build_training_node_snapshot,
    build_training_tree_snapshot,
)


class _DummyOverEvent:
    """Tiny over-event-like object with a stable string tag."""

    def get_over_tag(self) -> str:
        """Return a compact marker."""
        return "forced-win"


@dataclass(slots=True)
class _RichDummyNode:
    """Dummy node exposing the direct attribute surface used by the builder."""

    node_id: str
    parent_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    depth: int
    state: object
    direct_value: object | None
    backed_up_value: object | None
    is_terminal: bool
    is_exact: bool
    visit_count: int
    over_event: object | None = None


class _MinimalDummyNode:
    """Dummy node exposing only a stable node id."""

    node_id = "solo"


def test_build_training_tree_snapshot_with_dummy_nodes() -> None:
    """Builder should preserve ids, payloads, scalars, flags, and metadata."""
    root = _RichDummyNode(
        node_id="root",
        parent_ids=(),
        child_ids=("child",),
        depth=0,
        state={"raw_state": [1, 2, 3]},
        direct_value=1,
        backed_up_value=2.5,
        is_terminal=True,
        is_exact=True,
        visit_count=7,
        over_event=_DummyOverEvent(),
    )
    child = _RichDummyNode(
        node_id="child",
        parent_ids=("root",),
        child_ids=(),
        depth=1,
        state={"raw_state": [4]},
        direct_value=0.25,
        backed_up_value=None,
        is_terminal=False,
        is_exact=False,
        visit_count=2,
    )

    snapshot = build_training_tree_snapshot(
        [root, child],
        state_ref_dumper=lambda state: {"state_key": state},
        direct_value_extractor=lambda value: float(value) if value is not None else None,
        backed_up_value_extractor=lambda value: (
            float(value) if value is not None else None
        ),
        created_at_unix_s=123.5,
        metadata={"source": "unit-test"},
    )

    assert snapshot.root_node_id == "root"
    assert snapshot.created_at_unix_s == 123.5
    assert snapshot.metadata["format_kind"] == TRAINING_TREE_SNAPSHOT_FORMAT_KIND
    assert snapshot.metadata["format_version"] == (
        TRAINING_TREE_SNAPSHOT_FORMAT_VERSION
    )
    assert snapshot.metadata["source"] == "unit-test"
    assert snapshot.nodes[0].state_ref_payload == {
        "state_key": {"raw_state": [1, 2, 3]}
    }
    assert snapshot.nodes[0].direct_value_scalar == 1.0
    assert snapshot.nodes[0].backed_up_value_scalar == 2.5
    assert snapshot.nodes[0].is_terminal is True
    assert snapshot.nodes[0].is_exact is True
    assert snapshot.nodes[0].visit_count == 7
    assert snapshot.nodes[0].over_event_label == "forced-win"
    assert snapshot.nodes[1].parent_ids == ("root",)
    assert snapshot.nodes[1].child_ids == ()


def test_build_training_tree_snapshot_preserves_explicit_root_node_id() -> None:
    """Builder should keep an explicit root override unchanged."""
    root = _RichDummyNode(
        node_id="root",
        parent_ids=(),
        child_ids=(),
        depth=0,
        state={"raw_state": [1]},
        direct_value=None,
        backed_up_value=None,
        is_terminal=False,
        is_exact=False,
        visit_count=1,
    )

    snapshot = build_training_tree_snapshot([root], root_node_id="manual-root")

    assert snapshot.root_node_id == "manual-root"


def test_build_training_node_snapshot_handles_missing_optional_fields() -> None:
    """Builder should default missing optional fields cleanly."""
    snapshot = build_training_node_snapshot(_MinimalDummyNode())

    assert snapshot.node_id == "solo"
    assert snapshot.parent_ids == ()
    assert snapshot.child_ids == ()
    assert snapshot.depth == 0
    assert snapshot.state_ref_payload is None
    assert snapshot.direct_value_scalar is None
    assert snapshot.backed_up_value_scalar is None
    assert snapshot.is_terminal is False
    assert snapshot.is_exact is False
    assert snapshot.over_event_label is None
    assert snapshot.visit_count is None
    assert snapshot.metadata == {}


def test_build_training_node_snapshot_without_value_extractors_keeps_scalars_none() -> None:
    """Builder should leave scalar fields unset when no extractors are given."""
    node = _RichDummyNode(
        node_id="rich",
        parent_ids=(),
        child_ids=(),
        depth=2,
        state={"nested": {"board": [0, 1]}},
        direct_value=3,
        backed_up_value=4,
        is_terminal=False,
        is_exact=True,
        visit_count=9,
    )

    snapshot = build_training_node_snapshot(
        node,
        state_ref_dumper=lambda state: {"state_key": state},
    )

    assert snapshot.state_ref_payload == {
        "state_key": {"nested": {"board": [0, 1]}}
    }
    assert snapshot.direct_value_scalar is None
    assert snapshot.backed_up_value_scalar is None


def test_build_training_node_snapshot_keeps_structured_payloads_and_numeric_scalars() -> None:
    """Builder should keep payloads structured and value scalars numeric."""
    node = _RichDummyNode(
        node_id="rich",
        parent_ids=(),
        child_ids=(),
        depth=2,
        state={"nested": {"board": [0, 1]}},
        direct_value=3,
        backed_up_value=4,
        is_terminal=False,
        is_exact=True,
        visit_count=9,
    )

    snapshot = build_training_node_snapshot(
        node,
        state_ref_dumper=lambda state: {"state_key": state},
        direct_value_extractor=lambda value: float(value) if value is not None else None,
        backed_up_value_extractor=lambda value: (
            float(value) if value is not None else None
        ),
    )

    assert isinstance(snapshot.state_ref_payload, dict)
    assert snapshot.state_ref_payload == {
        "state_key": {"nested": {"board": [0, 1]}}
    }
    assert isinstance(snapshot.direct_value_scalar, float)
    assert isinstance(snapshot.backed_up_value_scalar, float)
    assert snapshot.direct_value_scalar == 3.0
    assert snapshot.backed_up_value_scalar == 4.0
