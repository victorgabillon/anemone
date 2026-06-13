"""Explicit value provenance snapshots and target selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from valanga.evaluations import Value  # noqa: TC002

from .value_candidate import ValueCandidate, ValueCandidateSource  # noqa: TC001


class NodeTargetSource(StrEnum):
    """Explicit source choices for node-level learning targets."""

    TREE_VALUE = "tree_value"
    EFFECTIVE_VALUE = "effective_value"
    DIRECT_VALUE = "direct_value"


class NodeValueSnapshotAccess(Protocol):
    """Minimal evaluation surface needed to snapshot value provenance."""

    @property
    def direct_value(self) -> Value | None:
        """Return the local evaluator value."""
        ...

    @property
    def tree_value(self) -> Value | None:
        """Return the child/subtree-derived value."""
        ...

    def get_effective_value_candidate(self) -> ValueCandidate:
        """Return the search-facing value with source provenance."""
        ...


@dataclass(frozen=True, slots=True)
class NodeValueSnapshot:
    """Direct, tree, and effective value views for one node evaluation."""

    direct_value: Value | None
    tree_value: Value | None
    effective_value: Value | None
    effective_value_source: ValueCandidateSource


def snapshot_node_values(node_eval: NodeValueSnapshotAccess) -> NodeValueSnapshot:
    """Return explicit direct/tree/effective value provenance for ``node_eval``."""
    effective = node_eval.get_effective_value_candidate()
    return NodeValueSnapshot(
        direct_value=node_eval.direct_value,
        tree_value=node_eval.tree_value,
        effective_value=effective.value,
        effective_value_source=effective.source,
    )


def select_node_target(
    node_eval: NodeValueSnapshotAccess,
    *,
    source: NodeTargetSource = NodeTargetSource.TREE_VALUE,
) -> Value | None:
    """Select an explicit node target value from a provenance-aware evaluation."""
    values = snapshot_node_values(node_eval)
    if source is NodeTargetSource.TREE_VALUE:
        return values.tree_value
    if source is NodeTargetSource.EFFECTIVE_VALUE:
        return values.effective_value
    if source is NodeTargetSource.DIRECT_VALUE:
        return values.direct_value
    raise ValueError(source)


__all__ = [
    "NodeTargetSource",
    "NodeValueSnapshot",
    "NodeValueSnapshotAccess",
    "select_node_target",
    "snapshot_node_values",
]
