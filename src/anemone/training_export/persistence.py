"""Persistence helpers for training-oriented tree snapshots.

The saved payload is JSON, so snapshot contents, especially
``state_ref_payload``, must already be JSON-serializable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._logging import _log_training_export_phase
from .serialization import (
    training_tree_snapshot_from_dict,
    training_tree_snapshot_to_dict,
)

if TYPE_CHECKING:
    from .model import TrainingTreeSnapshot


def save_training_tree_snapshot(
    snapshot: TrainingTreeSnapshot,
    path: str | Path,
) -> None:
    """Write ``snapshot`` as UTF-8 JSON to ``path``.

    Snapshot contents, especially ``state_ref_payload``, must be
    JSON-serializable before calling this helper.
    """
    target = Path(path)
    node_count = len(snapshot.nodes)
    with _log_training_export_phase(
        "total_save",
        node_count=node_count,
        output=target,
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        with _log_training_export_phase(
            "payload_to_dict",
            node_count=node_count,
            output=target,
        ):
            payload = training_tree_snapshot_to_dict(snapshot)
        with _log_training_export_phase(
            "json_dump",
            node_count=node_count,
            output=target,
        ):
            dumped_json = json.dumps(payload, indent=2) + "\n"
        encoded_json = dumped_json.encode("utf-8")
        with _log_training_export_phase(
            "json_write",
            node_count=node_count,
            output=target,
            bytes=len(encoded_json),
        ):
            target.write_bytes(encoded_json)


def load_training_tree_snapshot(
    path: str | Path,
) -> TrainingTreeSnapshot:
    """Load a training tree snapshot from ``path``."""
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    return training_tree_snapshot_from_dict(cast("dict[str, object]", loaded))


__all__ = [
    "load_training_tree_snapshot",
    "save_training_tree_snapshot",
]
