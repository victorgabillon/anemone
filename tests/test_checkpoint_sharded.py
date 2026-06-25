"""Tests for sharded checkpoint manifest primitives."""

from __future__ import annotations

import json
from random import Random
from typing import TYPE_CHECKING, Any

import pytest

from anemone.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    SHARDED_CHECKPOINT_FORMAT_VERSION,
    AlgorithmNodeCheckpointPayload,
    AnchorCheckpointStatePayload,
    LinkedChildCheckpointPayload,
    LinooCandidateCheckpointPayload,
    LinooCandidatesByDepthCheckpointPayload,
    LinooDepthStatsCheckpointPayload,
    LinooNodeStateCheckpointPayload,
    LinooSelectorCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    ShardedCheckpointManifest,
    ShardedCheckpointManifestError,
    ShardedCheckpointShardRef,
    TreeCheckpointPayload,
    TreeExpansionCheckpointPayload,
    TreeExpansionsCheckpointPayload,
    build_search_checkpoint_payload,
    load_checkpoint_json_payload,
    load_search_from_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    load_sharded_search_checkpoint,
    read_sharded_checkpoint_manifest,
    sharded_checkpoint_manifest_from_jsonable,
    sharded_checkpoint_manifest_to_jsonable,
    write_sharded_checkpoint_manifest,
    write_sharded_search_checkpoint,
)
from anemone.checkpoints.sharded import (
    _load_incremental_node_shells,
    _ShardedNodeShell,
)
from tests.fake_yaml_game import FakeYamlDynamics, MasterStateValueEvaluatorFromYaml
from tests.test_checkpoint_load import (
    _CHILDREN_BY_ID,
    _build_args,
    _build_runtime,
    _child_signature,
    _ConcreteFakeYamlState,
    _FakeIncrementalStateCheckpointCodec,
    _latest_expansion_signature,
    _parent_signature,
    _runtime_nodes_by_id,
    _serialized_value,
    _values_for,
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


def test_load_sharded_search_checkpoint_roundtrips_writer_payload(
    tmp_path: Path,
) -> None:
    """The C2c reader should reconstruct the C2b writer payload exactly."""
    payload = _small_checkpoint_payload()
    write_sharded_search_checkpoint(
        payload,
        tmp_path,
        node_count_per_shard=2,
        encoding="jsonl",
    )

    restored = load_sharded_search_checkpoint(tmp_path)

    assert restored == payload


def test_load_sharded_search_checkpoint_reads_node_shards_in_manifest_order(
    tmp_path: Path,
) -> None:
    """Multiple node-record shards should preserve the writer/manifest order."""
    payload = _small_checkpoint_payload()
    write_sharded_search_checkpoint(
        payload,
        tmp_path,
        node_count_per_shard=1,
        encoding="jsonl",
    )

    restored = load_sharded_search_checkpoint(tmp_path)

    assert [node.node_id for node in restored.tree.nodes] == [0, 1, 2]


def test_load_sharded_search_checkpoint_rejects_missing_metadata_shard(
    tmp_path: Path,
) -> None:
    """The reader should require exactly one metadata shard."""
    write_sharded_checkpoint_manifest(
        ShardedCheckpointManifest(
            total_node_count=0,
            shards=(
                ShardedCheckpointShardRef(
                    kind="node_records",
                    path="node_records/nodes_000000.jsonl",
                    record_count=0,
                    encoding="jsonl",
                ),
            ),
        ),
        tmp_path / "manifest.json",
    )

    with pytest.raises(ShardedCheckpointManifestError, match="metadata"):
        load_sharded_search_checkpoint(tmp_path)


def test_load_sharded_search_checkpoint_rejects_node_count_mismatch(
    tmp_path: Path,
) -> None:
    """Node-record counts should agree with the manifest total."""
    payload = _small_checkpoint_payload()
    manifest = write_sharded_search_checkpoint(
        payload,
        tmp_path,
        node_count_per_shard=2,
        encoding="jsonl",
    )
    write_sharded_checkpoint_manifest(
        ShardedCheckpointManifest(
            total_node_count=manifest.total_node_count + 1,
            total_branch_count=manifest.total_branch_count,
            shards=manifest.shards,
        ),
        tmp_path / "manifest.json",
    )

    with pytest.raises(ShardedCheckpointManifestError, match="node-record count"):
        load_sharded_search_checkpoint(tmp_path)


def test_load_sharded_search_checkpoint_rejects_unsupported_shard_kind(
    tmp_path: Path,
) -> None:
    """The C2c reader should reject shard kinds outside the C2b layout."""
    _write_raw_manifest(
        tmp_path,
        shards=[
            {
                "kind": "metadata",
                "path": "metadata.json.zst",
                "record_count": 1,
                "encoding": "json_zst",
            },
            {
                "kind": "state_payloads",
                "path": "state_payloads/state_000000.jsonl",
                "record_count": 0,
                "encoding": "jsonl",
            },
        ],
    )

    with pytest.raises(ShardedCheckpointManifestError, match="unsupported shard kind"):
        load_sharded_search_checkpoint(tmp_path)


def test_load_sharded_search_checkpoint_rejects_unsupported_shard_encoding(
    tmp_path: Path,
) -> None:
    """Node-record shards should be line-oriented JSON in C2c."""
    _write_raw_manifest(
        tmp_path,
        shards=[
            {
                "kind": "metadata",
                "path": "metadata.json.zst",
                "record_count": 1,
                "encoding": "json_zst",
            },
            {
                "kind": "node_records",
                "path": "node_records/nodes_000000.json",
                "record_count": 0,
                "encoding": "json",
            },
        ],
    )

    with pytest.raises(ShardedCheckpointManifestError, match="jsonl"):
        load_sharded_search_checkpoint(tmp_path)


def test_load_search_from_sharded_checkpoint_matches_payload_restore(
    tmp_path: Path,
) -> None:
    """C2d1 incremental sharded restore should match monolithic restore semantics."""
    runtime = _build_runtime()
    runtime.step()
    codec_for_build = _FakeIncrementalStateCheckpointCodec(_CHILDREN_BY_ID)
    payload = build_search_checkpoint_payload(runtime, state_codec=codec_for_build)
    write_sharded_search_checkpoint(
        payload,
        tmp_path,
        node_count_per_shard=2,
        encoding="jsonl",
    )

    monolithic = _restore_payload_for_sharded_test(
        payload,
        state_codec=_FakeIncrementalStateCheckpointCodec(_CHILDREN_BY_ID),
    )
    c2c_payload = load_sharded_search_checkpoint(tmp_path)
    c2c = _restore_payload_for_sharded_test(
        c2c_payload,
        state_codec=_FakeIncrementalStateCheckpointCodec(_CHILDREN_BY_ID),
    )
    incremental_codec = _FakeIncrementalStateCheckpointCodec(_CHILDREN_BY_ID)
    incremental_phases: list[str] = []
    incremental = load_search_from_sharded_checkpoint(
        tmp_path,
        state_codec=incremental_codec,
        dynamics=FakeYamlDynamics(),
        args=_build_args(),
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(
            _values_for(_CHILDREN_BY_ID)
        ),
        random_generator=Random(0),
        state_representation_factory=None,
        restore_memory_phase_logger=lambda phase, _metadata: incremental_phases.append(
            phase
        ),
    )

    _assert_runtime_equivalent(c2c, monolithic)
    _assert_runtime_equivalent(incremental, monolithic)
    assert incremental_codec.loaded_anchor_node_ids == []

    incremental_nodes = _runtime_nodes_by_id(incremental)
    monolithic_nodes = _runtime_nodes_by_id(monolithic)
    for node_id, incremental_node in incremental_nodes.items():
        assert incremental_node.state.node_id == monolithic_nodes[node_id].state.node_id

    assert incremental_phases == [
        "after_manifest_load",
        "after_state_payload_store_built",
        "after_nodes_created",
        "after_edges_and_runtime_state_restored",
        "after_selector_restored",
        "after_incremental_restore_done",
    ]


def test_incremental_node_shells_are_compact_not_full_node_payloads(
    tmp_path: Path,
) -> None:
    """Pass 1 of incremental restore should not retain all-node DTO shells."""
    payload = _small_checkpoint_payload()
    manifest = write_sharded_search_checkpoint(
        payload,
        tmp_path,
        node_count_per_shard=2,
        encoding="jsonl",
    )
    node_shards = [shard for shard in manifest.shards if shard.kind == "node_records"]

    (
        node_shells,
        state_payload_store,
        branch_count,
        anchor_count,
        delta_count,
        delta_parent_node_ids,
    ) = _load_incremental_node_shells(tmp_path, node_shards)

    assert len(node_shells) == len(payload.tree.nodes)
    assert all(isinstance(shell, _ShardedNodeShell) for shell in node_shells)
    assert not any(
        isinstance(shell, AlgorithmNodeCheckpointPayload) for shell in node_shells
    )
    assert not hasattr(node_shells[0], "state_payload")
    assert len(state_payload_store) == len(payload.tree.nodes)
    assert state_payload_store[0] == payload.tree.nodes[0].state_payload
    assert branch_count == manifest.total_branch_count
    assert anchor_count == len(payload.tree.nodes)
    assert delta_count == 0
    assert delta_parent_node_ids == []


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
        selector_state=LinooSelectorCheckpointPayload(
            depth_stats=[
                LinooDepthStatsCheckpointPayload(
                    depth=1,
                    total_nodes=2,
                    opened_count=0,
                    frontier_count=2,
                    terminal_count=0,
                    exact_count=0,
                    uncached_terminal_candidates=0,
                    non_openable_count=0,
                )
            ],
            node_states=[
                LinooNodeStateCheckpointPayload(
                    node_id=1,
                    depth=1,
                    status="frontier",
                )
            ],
            candidates_by_depth=[
                LinooCandidatesByDepthCheckpointPayload(
                    depth=1,
                    candidates=[
                        LinooCandidateCheckpointPayload(
                            node_id=1,
                            depth=1,
                            priority=0.5,
                            version=3,
                        )
                    ],
                )
            ],
            last_selected_node_id=1,
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


def _write_raw_manifest(
    tmp_path: Path,
    *,
    shards: list[dict[str, object]],
) -> None:
    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "sharded_checkpoint_format_version": SHARDED_CHECKPOINT_FORMAT_VERSION,
        "total_node_count": 0,
        "total_branch_count": 0,
        "generation": None,
        "node_count_per_shard": None,
        "encoder": "stdlib",
        "notes": None,
        "shards": shards,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _restore_payload_for_sharded_test(
    payload: SearchRuntimeCheckpointPayload,
    *,
    state_codec: _FakeIncrementalStateCheckpointCodec,
) -> Any:
    return load_search_from_checkpoint_payload(
        payload,
        state_codec=state_codec,
        dynamics=FakeYamlDynamics(),
        args=_build_args(),
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(
            _values_for(_CHILDREN_BY_ID)
        ),
        random_generator=Random(0),
        state_representation_factory=None,
    )


def _assert_runtime_equivalent(actual: Any, expected: Any) -> None:
    actual_nodes = _runtime_nodes_by_id(actual)
    expected_nodes = _runtime_nodes_by_id(expected)
    assert actual.tree.root_node.id == expected.tree.root_node.id
    assert actual.tree.nodes_count == expected.tree.nodes_count
    assert actual.tree.branch_count == expected.tree.branch_count
    assert set(actual_nodes) == set(expected_nodes)
    assert _latest_expansion_signature(actual) == _latest_expansion_signature(expected)

    for node_id, expected_node in expected_nodes.items():
        actual_node = actual_nodes[node_id]
        assert actual_node.tree_depth == expected_node.tree_depth
        assert _child_signature(actual_node) == _child_signature(expected_node)
        assert _parent_signature(actual_node) == _parent_signature(expected_node)
        assert _serialized_value(actual_node.tree_evaluation.direct_value) == (
            _serialized_value(expected_node.tree_evaluation.direct_value)
        )
        assert _serialized_value(actual_node.tree_evaluation.backed_up_value) == (
            _serialized_value(expected_node.tree_evaluation.backed_up_value)
        )
        assert actual_node.tree_evaluation.direct_evaluation_version == (
            expected_node.tree_evaluation.direct_evaluation_version
        )
