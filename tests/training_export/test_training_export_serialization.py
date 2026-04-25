"""Tests for training-export serialization and persistence."""
# ruff: noqa: E402

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_PACKAGE_ROOT = _REPO_ROOT / "src" / "anemone"

if "anemone" not in sys.modules:
    _stub_package = ModuleType("anemone")
    _stub_package.__path__ = [str(_SRC_PACKAGE_ROOT)]
    sys.modules["anemone"] = _stub_package

from anemone.training_export import (
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
    load_training_tree_snapshot,
    save_training_tree_snapshot,
    training_tree_snapshot_from_dict,
    training_tree_snapshot_to_dict,
)
from anemone.utils.logger import anemone_logger

if TYPE_CHECKING:
    from pathlib import Path as PathType


def _build_snapshot() -> TrainingTreeSnapshot:
    return TrainingTreeSnapshot(
        root_node_id="root",
        nodes=(
            TrainingNodeSnapshot(
                node_id="root",
                parent_ids=(),
                child_ids=("child",),
                depth=0,
                state_ref_payload={
                    "codec": "dummy",
                    "state": {"board": [1, 2], "turn": "WHITE"},
                },
                direct_value_scalar=0.5,
                backed_up_value_scalar=1.25,
                is_terminal=False,
                is_exact=True,
                over_event_label=None,
                visit_count=4,
                metadata={"node_kind": "root"},
            ),
            TrainingNodeSnapshot(
                node_id="child",
                parent_ids=("root",),
                child_ids=(),
                depth=1,
                state_ref_payload={"codec": "dummy", "state": {"board": [3]}},
                direct_value_scalar=None,
                backed_up_value_scalar=-1.0,
                is_terminal=True,
                is_exact=True,
                over_event_label="forced-loss",
                visit_count=None,
                metadata={},
            ),
        ),
        created_at_unix_s=12.5,
        metadata={
            "format_kind": "training_tree_snapshot",
            "format_version": 1,
            "source": "serialization-test",
        },
    )


def test_training_tree_snapshot_round_trip_through_dict() -> None:
    """Serialization helpers should preserve snapshots through dict payloads."""
    snapshot = _build_snapshot()

    payload = training_tree_snapshot_to_dict(snapshot)
    restored = training_tree_snapshot_from_dict(payload)

    assert restored == snapshot


def test_training_tree_snapshot_round_trip_through_disk(tmp_path: PathType) -> None:
    """Persistence helpers should preserve snapshots through disk round-trips."""
    path = tmp_path / "training_tree_snapshot.json"
    snapshot = _build_snapshot()

    save_training_tree_snapshot(snapshot, path)
    restored = load_training_tree_snapshot(path)

    assert restored == snapshot


def test_save_training_tree_snapshot_logs_structured_phases(
    tmp_path: PathType,
    caplog: object,
) -> None:
    """Persistence helper should emit structured training-export save logs."""
    path = tmp_path / "training_tree_snapshot.json"
    snapshot = _build_snapshot()

    with caplog.at_level(logging.INFO, logger=anemone_logger.name):
        save_training_tree_snapshot(snapshot, path)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "[training-export] phase=total_save status=start" in message
        and "node_count=2" in message
        for message in messages
    )
    assert any(
        "[training-export] phase=payload_to_dict status=done" in message
        and "elapsed_s=" in message
        for message in messages
    )
    assert any(
        "[training-export] phase=json_dump status=done" in message
        and "elapsed_s=" in message
        for message in messages
    )
    assert any(
        "[training-export] phase=json_write status=done" in message
        and "bytes=" in message
        for message in messages
    )
