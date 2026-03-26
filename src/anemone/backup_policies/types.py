"""Types used by backup policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from valanga import BranchKey

from anemone.node_evaluation.common.node_delta import NodeDelta


@dataclass(frozen=True, slots=True)
class BackupResult[BranchKeyT: BranchKey]:
    """Result flags produced by a backup policy execution."""

    value_changed: bool
    pv_changed: bool
    over_changed: bool
    node_delta: NodeDelta[BranchKeyT] = field(
        default_factory=cast("type[NodeDelta[BranchKeyT]]", NodeDelta)
    )
