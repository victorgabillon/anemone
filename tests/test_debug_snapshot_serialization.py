"""Tests for debug snapshot JSON serialization helpers."""

# pylint: disable=missing-function-docstring
# ruff: noqa: D103

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anemone.debug import (
    DebugEdgeView,
    DebugNodeView,
    DebugTreeSnapshot,
    export_snapshot_json,
    load_snapshot_json,
    snapshot_from_json,
    snapshot_to_json,
)

if TYPE_CHECKING:
    from pathlib import Path


def _build_snapshot() -> DebugTreeSnapshot:
    return DebugTreeSnapshot(
        nodes=(
            DebugNodeView(
                node_id="1",
                parent_ids=(),
                depth=0,
                label="id=1\ndepth=0",
                player_label="MAX",
                state_tag="root",
                direct_value="score=0.1",
                backed_up_value="score=0.9",
                principal_variation="a -> b",
                over_event="done",
                is_exact=True,
                is_terminal=False,
                index_fields={"index": "1.5"},
                child_ids=("2",),
                edge_labels_by_child={"2": "a"},
            ),
            DebugNodeView(
                node_id="2",
                parent_ids=("1",),
                depth=1,
                label="id=2\ndepth=1",
                child_ids=(),
            ),
        ),
        root_id="1",
        edges=(DebugEdgeView(parent_id="1", child_id="2", label="a"),),
    )


def test_snapshot_json_round_trip() -> None:
    snapshot = _build_snapshot()

    payload = snapshot_to_json(snapshot)
    restored = snapshot_from_json(payload)

    assert restored == snapshot


def test_export_snapshot_json_writes_readable_file(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.snapshot.json"
    snapshot = _build_snapshot()

    written_path = export_snapshot_json(snapshot, path)

    assert written_path == path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["root_id"] == "1"
    assert payload["nodes"][1]["node_id"] == "2"
    assert payload["edges"][0]["label"] == "a"
    assert payload["nodes"][0]["player_label"] == "MAX"
    assert payload["nodes"][0]["state_tag"] == "root"
    assert payload["nodes"][0]["direct_value"] == "score=0.1"
    assert payload["nodes"][0]["is_exact"] is True
    assert payload["nodes"][0]["is_terminal"] is False
    assert payload["nodes"][0]["edge_labels_by_child"] == {"2": "a"}
    assert load_snapshot_json(path) == snapshot


def test_snapshot_from_json_supports_minimal_legacy_payload() -> None:
    payload = {
        "root_id": "1",
        "nodes": [
            {
                "node_id": "1",
                "parent_ids": [],
                "depth": 0,
                "label": "id=1\ndepth=0",
            }
        ],
        "edges": [],
    }

    snapshot = snapshot_from_json(payload)

    assert snapshot.root_id == "1"
    assert snapshot.nodes[0] == DebugNodeView(
        node_id="1",
        parent_ids=(),
        depth=0,
        label="id=1\ndepth=0",
    )
