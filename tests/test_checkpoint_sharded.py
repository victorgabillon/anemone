"""Tests for sharded checkpoint manifest primitives."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from anemone.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    SHARDED_CHECKPOINT_FORMAT_VERSION,
    AlgorithmNodeCheckpointPayload,
    AnchorCheckpointStatePayload,
    LinkedChildCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    ShardedCheckpointManifest,
    ShardedCheckpointManifestError,
    ShardedCheckpointShardRef,
    TreeCheckpointPayload,
    TreeExpansionCheckpointPayload,
    TreeExpansionsCheckpointPayload,
    load_checkpoint_json_payload,
    read_sharded_checkpoint_manifest,
    sharded_checkpoint_manifest_from_jsonable,
    sharded_checkpoint_manifest_to_jsonable,
    write_sharded_checkpoint_manifest,
    write_sharded_search_checkpoint,
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


def test_write_sharded_search_checkpoint_creates_manifest_and_shards(
    tmp_path: Path,
) -> None:
    """The C2b writer should emit metadata, node shards, and global shards."""
    payload = _small_checkpoint_payload()

    manifest = write_sharded_search_checkpoint(
        payload,
        tmp_path,
        node_count_per_shard=2,
        encoding="jsonl",
        generation=3,
        encoder="stdlib",
    )

    restored_manifest = read_sharded_checkpoint_manifest(tmp_path / "manifest.json")
    assert restored_manifest == manifest
    assert manifest.total_node_count == 3
    assert manifest.total_branch_count == 2
    assert manifest.generation == 3
    assert manifest.node_count_per_shard == 2
    assert all(not shard.path.startswith("/") for shard in manifest.shards)
    assert all("\\" not in shard.path for shard in manifest.shards)
    for shard in manifest.shards:
        assert (tmp_path / shard.path).is_file()

    metadata_shard = _only_shard(manifest, "metadata")
    metadata_payload, _stats = load_checkpoint_json_payload(tmp_path / metadata_shard.path)
    assert metadata_payload["root_node_id"] == 0
    assert metadata_payload["node_count"] == 3
    assert metadata_payload["branch_count"] == 2

    latest_expansions_shard = _only_shard(manifest, "latest_expansions")
    latest_payload, _stats = load_checkpoint_json_payload(
        tmp_path / latest_expansions_shard.path
    )
    assert latest_payload["expansions_with_node_creation"][0]["child_node_id"] == 2


def test_write_sharded_search_checkpoint_node_shards_respect_target_size(
    tmp_path: Path,
) -> None:
    """Node-record shard counts should add up to the checkpoint node count."""
    payload = _small_checkpoint_payload()

    manifest = write_sharded_search_checkpoint(
        payload,
        tmp_path,
        node_count_per_shard=2,
        encoding="jsonl",
    )

    node_shards = [shard for shard in manifest.shards if shard.kind == "node_records"]
    assert [shard.record_count for shard in node_shards] == [2, 1]
    assert sum(shard.record_count for shard in node_shards) == manifest.total_node_count
    decoded_nodes = [
        record
        for shard in node_shards
        for record in _read_jsonl_records(tmp_path / shard.path)
    ]
    assert [node["node_id"] for node in decoded_nodes] == [0, 1, 2]
    assert decoded_nodes[0]["linked_children"][0]["child_node_id"] == 1
    assert decoded_nodes[2]["state_payload"]["anchor_ref"]["node_id"] == 2


def _small_checkpoint_payload() -> SearchRuntimeCheckpointPayload:
    return SearchRuntimeCheckpointPayload(
        evaluator_version=5,
        tree=TreeCheckpointPayload(
            root_node_id=0,
            nodes=[
                AlgorithmNodeCheckpointPayload(
                    node_id=0,
                    parent_node_id=None,
                    branch_from_parent=None,
                    depth=0,
                    state_payload=AnchorCheckpointStatePayload(
                        anchor_ref={"node_id": 0}
                    ),
                    generated_all_branches=True,
                    linked_children=[
                        LinkedChildCheckpointPayload(branch_key="a", child_node_id=1),
                        LinkedChildCheckpointPayload(branch_key="b", child_node_id=2),
                    ],
                ),
                AlgorithmNodeCheckpointPayload(
                    node_id=1,
                    parent_node_id=0,
                    branch_from_parent="a",
                    depth=1,
                    state_payload=AnchorCheckpointStatePayload(
                        anchor_ref={"node_id": 1}
                    ),
                    generated_all_branches=False,
                ),
                AlgorithmNodeCheckpointPayload(
                    node_id=2,
                    parent_node_id=0,
                    branch_from_parent="b",
                    depth=1,
                    state_payload=AnchorCheckpointStatePayload(
                        anchor_ref={"node_id": 2}
                    ),
                    generated_all_branches=False,
                ),
            ],
        ),
        latest_tree_expansions=TreeExpansionsCheckpointPayload(
            expansions_with_node_creation=[
                TreeExpansionCheckpointPayload(
                    child_node_id=2,
                    parent_node_id=0,
                    branch_key="b",
                    creation_child_node=True,
                )
            ]
        ),
    )


def _only_shard(
    manifest: ShardedCheckpointManifest,
    kind: str,
) -> ShardedCheckpointShardRef:
    matches = [shard for shard in manifest.shards if shard.kind == kind]
    assert len(matches) == 1
    return matches[0]


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
