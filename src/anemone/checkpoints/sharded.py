"""Compatibility façade for sharded checkpoint APIs."""

from __future__ import annotations

from .sharded_decode import load_sharded_search_checkpoint
from .sharded_encode import write_sharded_search_checkpoint
from .sharded_errors import ShardedCheckpointManifestError
from .sharded_manifest import (
    read_sharded_checkpoint_manifest,
    sharded_checkpoint_manifest_from_jsonable,
    sharded_checkpoint_manifest_to_jsonable,
    write_sharded_checkpoint_manifest,
)
from .sharded_restore import load_search_from_sharded_checkpoint
from .sharded_types import (
    SHARDED_CHECKPOINT_FORMAT_VERSION,
    ShardedCheckpointManifest,
    ShardedCheckpointShardEncoding,
    ShardedCheckpointShardKind,
    ShardedCheckpointShardRef,
)

__all__ = [
    "SHARDED_CHECKPOINT_FORMAT_VERSION",
    "ShardedCheckpointManifest",
    "ShardedCheckpointManifestError",
    "ShardedCheckpointShardEncoding",
    "ShardedCheckpointShardKind",
    "ShardedCheckpointShardRef",
    "load_search_from_sharded_checkpoint",
    "load_sharded_search_checkpoint",
    "read_sharded_checkpoint_manifest",
    "sharded_checkpoint_manifest_from_jsonable",
    "sharded_checkpoint_manifest_to_jsonable",
    "write_sharded_checkpoint_manifest",
    "write_sharded_search_checkpoint",
]
