"""Shared types for sharded checkpoint manifests and shards."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

from .payloads import CHECKPOINT_FORMAT_VERSION
from .sharded_errors import ShardedCheckpointManifestError

if TYPE_CHECKING:
    from .io import CheckpointJsonEncoder

SHARDED_CHECKPOINT_FORMAT_VERSION = 1
_DEFAULT_NODE_RUNTIME_RESTORE_BATCH_SIZE = 2048

type ShardedCheckpointShardKind = Literal[
    "metadata",
    "tree_nodes",
    "node_records",
    "node_shells",
    "state_payloads",
    "node_runtime",
    "linked_children",
    "node_evaluations",
    "selector",
    "latest_expansions",
]
type ShardedCheckpointShardEncoding = Literal["json", "json_zst", "jsonl", "jsonl_zst"]
type ShardedCheckpointLayout = Literal["node_records", "split"]


def _validate_relative_manifest_path(path: str) -> None:
    """Require portable relative POSIX paths inside sharded manifests."""
    if not path:
        raise ShardedCheckpointManifestError.empty_path()
    if "\\" in path:
        raise ShardedCheckpointManifestError.non_posix_path()
    posix_path = PurePosixPath(path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ShardedCheckpointManifestError.unsafe_path()
    if any(part in {"", "."} for part in posix_path.parts):
        raise ShardedCheckpointManifestError.unnormalized_path()


@dataclass(frozen=True, slots=True)
class ShardedCheckpointShardRef:
    """Reference to one relative shard file within a sharded checkpoint directory."""

    kind: ShardedCheckpointShardKind
    path: str
    record_count: int
    encoding: ShardedCheckpointShardEncoding
    compressed_bytes: int | None = None
    uncompressed_bytes: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate stable, portable shard metadata."""
        _validate_relative_manifest_path(self.path)
        if self.record_count < 0:
            raise ShardedCheckpointManifestError.negative_shard_record_count()
        if self.compressed_bytes is not None and self.compressed_bytes < 0:
            raise ShardedCheckpointManifestError.negative_shard_compressed_bytes()
        if self.uncompressed_bytes is not None and self.uncompressed_bytes < 0:
            raise ShardedCheckpointManifestError.negative_shard_uncompressed_bytes()


@dataclass(frozen=True, slots=True)
class ShardedCheckpointManifest:
    """Top-level manifest for a generic sharded Anemone runtime checkpoint."""

    checkpoint_format_version: int = CHECKPOINT_FORMAT_VERSION
    sharded_checkpoint_format_version: int = SHARDED_CHECKPOINT_FORMAT_VERSION
    total_node_count: int = 0
    total_branch_count: int | None = None
    generation: int | None = None
    node_count_per_shard: int | None = None
    shards: tuple[ShardedCheckpointShardRef, ...] = field(default_factory=tuple)
    encoder: CheckpointJsonEncoder | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Normalize shard references and validate counts."""
        if self.checkpoint_format_version != CHECKPOINT_FORMAT_VERSION:
            raise ShardedCheckpointManifestError.unsupported_checkpoint_format_version()
        if self.sharded_checkpoint_format_version != SHARDED_CHECKPOINT_FORMAT_VERSION:
            raise ShardedCheckpointManifestError.unsupported_sharded_format_version()
        if self.total_node_count < 0:
            raise ShardedCheckpointManifestError.negative_total_node_count()
        if self.total_branch_count is not None and self.total_branch_count < 0:
            raise ShardedCheckpointManifestError.negative_total_branch_count()
        if self.node_count_per_shard is not None and self.node_count_per_shard <= 0:
            raise ShardedCheckpointManifestError.non_positive_node_count_per_shard()
        object.__setattr__(self, "shards", tuple(self.shards))
