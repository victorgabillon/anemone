"""Tests for sharded checkpoint manifest primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from anemone.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    SHARDED_CHECKPOINT_FORMAT_VERSION,
    ShardedCheckpointManifest,
    ShardedCheckpointManifestError,
    ShardedCheckpointShardRef,
    read_sharded_checkpoint_manifest,
    sharded_checkpoint_manifest_from_jsonable,
    sharded_checkpoint_manifest_to_jsonable,
    write_sharded_checkpoint_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_sharded_checkpoint_manifest_json_roundtrip(tmp_path: Path) -> None:
    """Manifest helpers should preserve stable shard metadata."""
    manifest = ShardedCheckpointManifest(
        total_node_count=42,
        total_branch_count=50,
        generation=7,
        node_count_per_shard=1000,
        encoder="orjson",
        shards=(
            ShardedCheckpointShardRef(
                kind="metadata",
                path="metadata.json.zst",
                record_count=1,
                encoding="json_zst",
                compressed_bytes=123,
                uncompressed_bytes=456,
                sha256="abc",
            ),
            ShardedCheckpointShardRef(
                kind="tree_nodes",
                path="tree_nodes/nodes_000000.jsonl.zst",
                record_count=42,
                encoding="jsonl_zst",
            ),
        ),
    )
    manifest_path = tmp_path / "manifest.json"

    written_path = write_sharded_checkpoint_manifest(manifest, manifest_path)
    restored = read_sharded_checkpoint_manifest(written_path)

    assert restored == manifest
    jsonable = sharded_checkpoint_manifest_to_jsonable(restored)
    assert jsonable["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert (
        jsonable["sharded_checkpoint_format_version"]
        == SHARDED_CHECKPOINT_FORMAT_VERSION
    )
    assert jsonable["shards"][1]["path"] == "tree_nodes/nodes_000000.jsonl.zst"


@pytest.mark.parametrize(
    "bad_path",
    [
        "/absolute/shard.jsonl.zst",
        "../escape.jsonl.zst",
        "tree_nodes/../../escape.jsonl.zst",
        r"tree_nodes\nodes_000000.jsonl.zst",
        "",
    ],
)
def test_sharded_checkpoint_manifest_rejects_unsafe_shard_paths(bad_path: str) -> None:
    """Shard paths must be stable relative paths inside the checkpoint directory."""
    with pytest.raises(ShardedCheckpointManifestError):
        ShardedCheckpointShardRef(
            kind="tree_nodes",
            path=bad_path,
            record_count=1,
            encoding="jsonl_zst",
        )


def test_sharded_checkpoint_manifest_from_jsonable_rejects_non_list_shards() -> None:
    """Decoded manifest validation should reject malformed shard containers."""
    with pytest.raises(ShardedCheckpointManifestError):
        sharded_checkpoint_manifest_from_jsonable(
            {
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "sharded_checkpoint_format_version": (
                    SHARDED_CHECKPOINT_FORMAT_VERSION
                ),
                "total_node_count": 0,
                "shards": {},
            }
        )
