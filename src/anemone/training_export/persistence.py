"""Persistence helpers for training-oriented tree snapshots.

The saved payload is JSON, so snapshot contents, especially
``state_ref_payload``, must already be JSON-serializable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

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
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(training_tree_snapshot_to_dict(snapshot), indent=2) + "\n",
        encoding="utf-8",
    )


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
