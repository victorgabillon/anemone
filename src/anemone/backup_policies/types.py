"""Types used by backup policies."""

from __future__ import annotations

from dataclasses import dataclass, field

from valanga import BranchKey

from anemone.node_evaluation.common.node_delta import NodeDelta


@dataclass(frozen=True, slots=True)
class BackupResult:
    """Result flags produced by a backup policy execution."""

    value_changed: bool
    pv_changed: bool
    over_changed: bool
    node_delta: NodeDelta[BranchKey] = field(default_factory=NodeDelta)
