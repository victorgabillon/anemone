"""Manifest JSON serialization for sharded checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._json_types import (
    optional_int_field,
    optional_str_field,
    require_int,
    require_int_field,
    require_str_field,
)
from .payloads import CHECKPOINT_FORMAT_VERSION
from .sharded_errors import ShardedCheckpointManifestError
from .sharded_types import (
    SHARDED_CHECKPOINT_FORMAT_VERSION,
    ShardedCheckpointManifest,
    ShardedCheckpointShardRef,
)

if TYPE_CHECKING:
    from .io import CheckpointJsonEncoder
    from .sharded_types import (
        ShardedCheckpointShardEncoding,
        ShardedCheckpointShardKind,
    )


def sharded_checkpoint_manifest_to_jsonable(
    manifest: ShardedCheckpointManifest,
) -> dict[str, object]:
    """Return a stable JSON-ready representation of one sharded manifest."""
    return {
        "checkpoint_format_version": manifest.checkpoint_format_version,
        "sharded_checkpoint_format_version": (
            manifest.sharded_checkpoint_format_version
        ),
        "total_node_count": manifest.total_node_count,
        "total_branch_count": manifest.total_branch_count,
        "generation": manifest.generation,
        "node_count_per_shard": manifest.node_count_per_shard,
        "encoder": manifest.encoder,
        "notes": manifest.notes,
        "shards": [
            {
                "kind": shard.kind,
                "path": shard.path,
                "record_count": shard.record_count,
                "encoding": shard.encoding,
                "compressed_bytes": shard.compressed_bytes,
                "uncompressed_bytes": shard.uncompressed_bytes,
                "sha256": shard.sha256,
            }
            for shard in manifest.shards
        ],
    }


def sharded_checkpoint_manifest_from_jsonable(
    payload: object,
) -> ShardedCheckpointManifest:
    """Build and validate a sharded manifest from decoded JSON data."""
    if not isinstance(payload, dict):
        raise ShardedCheckpointManifestError.manifest_not_mapping()
    data = cast("dict[str, object]", payload)
    raw_shards = data.get("shards", [])
    if not isinstance(raw_shards, list):
        raise ShardedCheckpointManifestError.shards_not_list()
    raw_shard_payloads = cast("list[object]", raw_shards)
    return ShardedCheckpointManifest(
        checkpoint_format_version=require_int(
            data.get("checkpoint_format_version", CHECKPOINT_FORMAT_VERSION),
            field_name="checkpoint_format_version",
        ),
        sharded_checkpoint_format_version=require_int(
            data.get(
                "sharded_checkpoint_format_version",
                SHARDED_CHECKPOINT_FORMAT_VERSION,
            ),
            field_name="sharded_checkpoint_format_version",
        ),
        total_node_count=require_int(
            data.get("total_node_count", 0),
            field_name="total_node_count",
        ),
        total_branch_count=optional_int_field(data, "total_branch_count"),
        generation=optional_int_field(data, "generation"),
        node_count_per_shard=optional_int_field(data, "node_count_per_shard"),
        encoder=cast(
            "CheckpointJsonEncoder | None", optional_str_field(data, "encoder")
        ),
        notes=optional_str_field(data, "notes"),
        shards=tuple(
            _sharded_checkpoint_shard_ref_from_jsonable(raw_shard)
            for raw_shard in raw_shard_payloads
        ),
    )


def write_sharded_checkpoint_manifest(
    manifest: ShardedCheckpointManifest,
    output_path: str | Path,
) -> Path:
    """Write one sharded checkpoint manifest JSON file."""
    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(
            sharded_checkpoint_manifest_to_jsonable(manifest),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return resolved_output


def read_sharded_checkpoint_manifest(
    input_path: str | Path,
) -> ShardedCheckpointManifest:
    """Read and validate one sharded checkpoint manifest JSON file."""
    with Path(input_path).open(encoding="utf-8") as handle:
        return sharded_checkpoint_manifest_from_jsonable(json.load(handle))


def _sharded_checkpoint_shard_ref_from_jsonable(
    payload: object,
) -> ShardedCheckpointShardRef:
    if not isinstance(payload, dict):
        raise ShardedCheckpointManifestError.shard_not_mapping()
    data = cast("dict[str, object]", payload)
    return ShardedCheckpointShardRef(
        kind=cast("ShardedCheckpointShardKind", require_str_field(data, "kind")),
        path=require_str_field(data, "path"),
        record_count=require_int_field(data, "record_count"),
        encoding=cast(
            "ShardedCheckpointShardEncoding",
            require_str_field(data, "encoding"),
        ),
        compressed_bytes=optional_int_field(data, "compressed_bytes"),
        uncompressed_bytes=optional_int_field(data, "uncompressed_bytes"),
        sha256=optional_str_field(data, "sha256"),
    )
