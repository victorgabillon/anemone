"""Typed payloads for Anemone search-runtime checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class EnumMemberPayload:
    """Serializable reference to one Enum member."""

    module: str
    qualname: str
    name: str


type CheckpointAtomPayload = None | bool | int | float | str | EnumMemberPayload


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


@dataclass(slots=True)
class DecisionOrderingCheckpointPayload:
    """Serialized decision-ordering cache for one node."""

    branch_ordering: list[BranchOrderingCheckpointPayload] = field(
        default_factory=list
    )


@dataclass(slots=True)
class PrincipalVariationCheckpointPayload:
    """Serialized principal-variation state for one node."""

    best_branch_sequence: list[CheckpointAtomPayload] = field(default_factory=list)
    pv_version: int = 0
    cached_best_child_version: int | None = None


@dataclass(slots=True)
class BranchFrontierCheckpointPayload:
    """Serialized branch-frontier membership for one node."""

    frontier_branches: list[CheckpointAtomPayload] = field(default_factory=list)


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
    unopened_branches: list[CheckpointAtomPayload] = field(default_factory=list)
    linked_children_by_branch: dict[CheckpointAtomPayload, int] = field(
        default_factory=dict
    )
    evaluation: NodeEvaluationCheckpointPayload | None = None
    exploration_index: ExplorationIndexCheckpointPayload | None = None


@dataclass(slots=True)
class TreeCheckpointPayload:
    """Serialized structural/search payload for one tree."""

    root_node_id: int
    nodes: list[AlgorithmNodeCheckpointPayload] = field(default_factory=list)


@dataclass(slots=True)
class SearchRuntimeCheckpointPayload:
    """Top-level runtime checkpoint payload for a search session."""

    evaluator_version: int
    tree: TreeCheckpointPayload
    format_version: int = CHECKPOINT_FORMAT_VERSION
    rng_state: object | None = None
    latest_tree_expansions: object | None = None


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "AlgorithmNodeCheckpointPayload",
    "BackupRuntimeCheckpointPayload",
    "BranchFrontierCheckpointPayload",
    "BranchOrderingCheckpointPayload",
    "CheckpointAtomPayload",
    "DecisionOrderingCheckpointPayload",
    "EnumMemberPayload",
    "ExplorationIndexCheckpointPayload",
    "NodeEvaluationCheckpointPayload",
    "PrincipalVariationCheckpointPayload",
    "SearchRuntimeCheckpointPayload",
    "SerializedOverEventPayload",
    "SerializedValuePayload",
    "TreeCheckpointPayload",
]
