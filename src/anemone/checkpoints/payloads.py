"""Typed payloads for Anemone search-runtime checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

CHECKPOINT_FORMAT_VERSION = 1


class EnumAtomPayload(TypedDict):
    """Explicit tagged payload that preserves Enum members across checkpoints."""

    type: Literal["enum"]
    module: str
    qualname: str
    name: str


class TupleAtomPayload(TypedDict):
    """Explicit tagged payload that preserves tuple atoms across checkpoints."""

    type: Literal["tuple"]
    items: list[CheckpointAtomPayload]


type CheckpointAtomPayload = (
    None | bool | int | float | str | EnumAtomPayload | TupleAtomPayload
)


def _empty_checkpoint_atoms() -> list[CheckpointAtomPayload]:
    """Return a typed empty checkpoint-atom list."""
    return []


@dataclass(frozen=True, slots=True)
class SerializedOverEventPayload:
    """Serialized terminal-outcome metadata carried by a ``Value``."""

    outcome: str
    termination: CheckpointAtomPayload = None
    winner: CheckpointAtomPayload = None


@dataclass(frozen=True, slots=True)
class SerializedValuePayload:
    """Serialized checkpoint form of a ``valanga.evaluations.Value``."""

    score: float
    certainty: str
    over_event: SerializedOverEventPayload | None = None
    line: list[CheckpointAtomPayload] | None = None


@dataclass(frozen=True, slots=True)
class BranchOrderingCheckpointPayload:
    """Serialized branch-ordering cache entry for one child branch."""

    branch_key: CheckpointAtomPayload
    primary_score: float
    tactical_tiebreak: int
    stable_tiebreak_id: int


def _empty_branch_ordering_payloads() -> list[BranchOrderingCheckpointPayload]:
    """Return a typed empty branch-ordering payload list."""
    return []


@dataclass(slots=True)
class DecisionOrderingCheckpointPayload:
    """Serialized decision-ordering cache for one node."""

    branch_ordering: list[BranchOrderingCheckpointPayload] = field(
        default_factory=_empty_branch_ordering_payloads
    )


@dataclass(slots=True)
class PrincipalVariationCheckpointPayload:
    """Serialized principal-variation state for one node."""

    best_branch_sequence: list[CheckpointAtomPayload] = field(
        default_factory=_empty_checkpoint_atoms
    )
    pv_version: int = 0
    cached_best_child_version: int | None = None


@dataclass(slots=True)
class BranchFrontierCheckpointPayload:
    """Serialized branch-frontier membership for one node."""

    frontier_branches: list[CheckpointAtomPayload] = field(
        default_factory=_empty_checkpoint_atoms
    )


@dataclass(slots=True)
class BackupRuntimeCheckpointPayload:
    """Serialized conservative backup-runtime cache for one node."""

    best_branch: CheckpointAtomPayload | None = None
    second_best_branch: CheckpointAtomPayload | None = None
    exact_child_count: int = 0
    selected_child_pv_version: int | None = None
    is_initialized: bool = False


@dataclass(slots=True)
class NodeEvaluationCheckpointPayload:
    """Serialized value/evaluation runtime attached to one node."""

    direct_value: SerializedValuePayload | None = None
    direct_evaluation_version: int | None = None
    backed_up_value: SerializedValuePayload | None = None
    decision_ordering: DecisionOrderingCheckpointPayload | None = None
    principal_variation: PrincipalVariationCheckpointPayload | None = None
    branch_frontier: BranchFrontierCheckpointPayload | None = None
    backup_runtime: BackupRuntimeCheckpointPayload | None = None


@dataclass(slots=True)
class ExplorationIndexCheckpointPayload:
    """Serialized placeholder for exploration-index runtime state."""

    kind: str | None = None
    payload: object | None = None


@dataclass(frozen=True, slots=True)
class LinkedChildCheckpointPayload:
    """Serialized parent-child edge keyed by one checkpoint atom payload."""

    branch_key: CheckpointAtomPayload
    child_node_id: int


def _empty_linked_children() -> list[LinkedChildCheckpointPayload]:
    """Return a typed empty linked-child payload list."""
    return []


@dataclass(slots=True)
class AlgorithmNodeCheckpointPayload:
    """Serialized runtime/search payload for one algorithm node.

    ``state_ref`` is opaque checkpoint data supplied by the domain-side state
    checkpoint codec. Anemone stores it but does not interpret it.
    """

    node_id: int
    parent_node_id: int | None
    branch_from_parent: CheckpointAtomPayload | None
    depth: int
    state_ref: object
    generated_all_branches: bool
    unopened_branches: list[CheckpointAtomPayload] = field(
        default_factory=_empty_checkpoint_atoms
    )
    linked_children: list[LinkedChildCheckpointPayload] = field(
        default_factory=_empty_linked_children
    )
    evaluation: NodeEvaluationCheckpointPayload | None = None
    exploration_index: ExplorationIndexCheckpointPayload | None = None


def _empty_algorithm_node_payloads() -> list[AlgorithmNodeCheckpointPayload]:
    """Return a typed empty algorithm-node payload list."""
    return []


@dataclass(slots=True)
class TreeCheckpointPayload:
    """Serialized structural/search payload for one tree."""

    root_node_id: int
    nodes: list[AlgorithmNodeCheckpointPayload] = field(
        default_factory=_empty_algorithm_node_payloads
    )


@dataclass(slots=True)
class TreeExpansionCheckpointPayload:
    """Serialized selector-visible record for one recent tree expansion."""

    child_node_id: int
    parent_node_id: int | None
    branch_key: CheckpointAtomPayload | None
    creation_child_node: bool


def _empty_tree_expansion_payloads() -> list[TreeExpansionCheckpointPayload]:
    """Return a typed empty tree-expansion payload list."""
    return []


@dataclass(slots=True)
class TreeExpansionsCheckpointPayload:
    """Serialized selector-visible expansion records from the latest iteration."""

    expansions_with_node_creation: list[TreeExpansionCheckpointPayload] = field(
        default_factory=_empty_tree_expansion_payloads
    )
    expansions_without_node_creation: list[TreeExpansionCheckpointPayload] = field(
        default_factory=_empty_tree_expansion_payloads
    )


@dataclass(slots=True)
class SearchRuntimeCheckpointPayload:
    """Top-level runtime checkpoint payload for a search session."""

    evaluator_version: int
    tree: TreeCheckpointPayload
    format_version: int = CHECKPOINT_FORMAT_VERSION
    rng_state: object | None = None
    latest_tree_expansions: TreeExpansionsCheckpointPayload | None = None


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "AlgorithmNodeCheckpointPayload",
    "BackupRuntimeCheckpointPayload",
    "BranchFrontierCheckpointPayload",
    "BranchOrderingCheckpointPayload",
    "CheckpointAtomPayload",
    "DecisionOrderingCheckpointPayload",
    "EnumAtomPayload",
    "ExplorationIndexCheckpointPayload",
    "LinkedChildCheckpointPayload",
    "NodeEvaluationCheckpointPayload",
    "PrincipalVariationCheckpointPayload",
    "SearchRuntimeCheckpointPayload",
    "SerializedOverEventPayload",
    "SerializedValuePayload",
    "TreeCheckpointPayload",
    "TreeExpansionCheckpointPayload",
    "TreeExpansionsCheckpointPayload",
    "TupleAtomPayload",
]
