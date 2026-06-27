"""Write sharded checkpoint payloads to manifest and shard files."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from pathlib import Path
from typing import TYPE_CHECKING

from .sharded_errors import ShardedCheckpointManifestError
from .sharded_io import _write_json_shard, _write_jsonl_record_shards
from .sharded_manifest import write_sharded_checkpoint_manifest
from .sharded_types import (
    SHARDED_CHECKPOINT_FORMAT_VERSION,
    ShardedCheckpointLayout,
    ShardedCheckpointManifest,
    ShardedCheckpointShardEncoding,
    ShardedCheckpointShardRef,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .io import CheckpointJsonEncoder
    from .payloads import AlgorithmNodeCheckpointPayload, SearchRuntimeCheckpointPayload


def write_sharded_search_checkpoint(
    payload: SearchRuntimeCheckpointPayload,
    output_dir: str | Path,
    *,
    node_count_per_shard: int = 100_000,
    encoding: ShardedCheckpointShardEncoding = "jsonl_zst",
    layout: ShardedCheckpointLayout = "node_records",
    generation: int | None = None,
    encoder: CheckpointJsonEncoder | None = None,
) -> ShardedCheckpointManifest:
    """Write a generic sharded runtime checkpoint."""
    if node_count_per_shard <= 0:
        raise ShardedCheckpointManifestError.non_positive_node_count_per_shard()
    if encoding not in {"jsonl", "jsonl_zst"}:
        raise ShardedCheckpointManifestError.unsupported_jsonl_encoding()
    if layout not in {"node_records", "split"}:
        raise ShardedCheckpointManifestError.unsupported_layout()
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_encoder: CheckpointJsonEncoder = "stdlib" if encoder is None else encoder

    shards: list[ShardedCheckpointShardRef] = []
    metadata = {
        "format_version": payload.format_version,
        "evaluator_version": payload.evaluator_version,
        "rng_state": payload.rng_state,
        "root_node_id": payload.tree.root_node_id,
        "node_count": len(payload.tree.nodes),
        "branch_count": _checkpoint_branch_count(payload),
    }
    shards.append(
        _write_json_shard(
            metadata,
            output_dir=resolved_output_dir,
            relative_path="metadata.json.zst",
            kind="metadata",
        )
    )
    if layout == "split":
        shards.extend(
            _write_split_node_shards(
                payload.tree.nodes,
                output_dir=resolved_output_dir,
                node_count_per_shard=node_count_per_shard,
                encoding=encoding,
                encoder=resolved_encoder,
            )
        )
    else:
        shards.extend(
            _write_node_record_shards(
                payload.tree.nodes,
                output_dir=resolved_output_dir,
                node_count_per_shard=node_count_per_shard,
                encoding=encoding,
                encoder=resolved_encoder,
            )
        )
    if payload.selector_state is not None:
        shards.append(
            _write_json_shard(
                payload.selector_state,
                output_dir=resolved_output_dir,
                relative_path="selector/selector.json.zst",
                kind="selector",
            )
        )
    if payload.latest_tree_expansions is not None:
        shards.append(
            _write_json_shard(
                payload.latest_tree_expansions,
                output_dir=resolved_output_dir,
                relative_path="latest_expansions/latest_expansions.json.zst",
                kind="latest_expansions",
            )
        )

    manifest = ShardedCheckpointManifest(
        checkpoint_format_version=payload.format_version,
        sharded_checkpoint_format_version=SHARDED_CHECKPOINT_FORMAT_VERSION,
        total_node_count=len(payload.tree.nodes),
        total_branch_count=_checkpoint_branch_count(payload),
        generation=generation,
        node_count_per_shard=node_count_per_shard,
        shards=tuple(shards),
        encoder=resolved_encoder,
        notes=(
            "C2e split state/node/runtime shard layout."
            if layout == "split"
            else "C2b first-stage node-record shard layout."
        ),
    )
    write_sharded_checkpoint_manifest(manifest, resolved_output_dir / "manifest.json")
    return manifest


def _checkpoint_branch_count(payload: SearchRuntimeCheckpointPayload) -> int:
    """Return the number of linked child edges in one checkpoint payload."""
    return sum(len(node_payload.linked_children) for node_payload in payload.tree.nodes)


def _write_node_record_shards(
    node_payloads: Iterable[object],
    *,
    output_dir: Path,
    node_count_per_shard: int,
    encoding: ShardedCheckpointShardEncoding,
    encoder: CheckpointJsonEncoder,
) -> list[ShardedCheckpointShardRef]:
    """Write node-record JSONL shards and return manifest refs."""
    return _write_jsonl_record_shards(
        node_payloads,
        output_dir=output_dir,
        node_count_per_shard=node_count_per_shard,
        encoding=encoding,
        encoder=encoder,
        kind="node_records",
        directory_name="node_records",
        file_stem="nodes",
    )


def _write_split_node_shards(
    node_payloads: Iterable[AlgorithmNodeCheckpointPayload],
    *,
    output_dir: Path,
    node_count_per_shard: int,
    encoding: ShardedCheckpointShardEncoding,
    encoder: CheckpointJsonEncoder,
) -> list[ShardedCheckpointShardRef]:
    """Write split node shell, state payload, and runtime JSONL shard families."""
    payloads = tuple(node_payloads)
    shards: list[ShardedCheckpointShardRef] = []
    shards.extend(
        _write_jsonl_record_shards(
            (_node_shell_record(node_payload) for node_payload in payloads),
            output_dir=output_dir,
            node_count_per_shard=node_count_per_shard,
            encoding=encoding,
            encoder=encoder,
            kind="node_shells",
            directory_name="node_shells",
            file_stem="nodes",
        )
    )
    shards.extend(
        _write_jsonl_record_shards(
            (_state_payload_record(node_payload) for node_payload in payloads),
            output_dir=output_dir,
            node_count_per_shard=node_count_per_shard,
            encoding=encoding,
            encoder=encoder,
            kind="state_payloads",
            directory_name="state_payloads",
            file_stem="state_payloads",
        )
    )
    shards.extend(
        _write_jsonl_record_shards(
            (_node_runtime_record(node_payload) for node_payload in payloads),
            output_dir=output_dir,
            node_count_per_shard=node_count_per_shard,
            encoding=encoding,
            encoder=encoder,
            kind="node_runtime",
            directory_name="node_runtime",
            file_stem="node_runtime",
        )
    )
    return shards


def _node_shell_record(
    node_payload: AlgorithmNodeCheckpointPayload,
) -> dict[str, object]:
    """Return split-layout compact node metadata for one node."""
    return {
        "node_id": node_payload.node_id,
        "parent_node_id": node_payload.parent_node_id,
        "branch_from_parent": node_payload.branch_from_parent,
        "depth": node_payload.depth,
        "generated_all_branches": node_payload.generated_all_branches,
        "state_summary": node_payload.state_payload.state_summary,
    }


def _state_payload_record(
    node_payload: AlgorithmNodeCheckpointPayload,
) -> dict[str, object]:
    """Return split-layout state payload record for one node."""
    return {
        "node_id": node_payload.node_id,
        "state_payload": node_payload.state_payload,
    }


def _node_runtime_record(
    node_payload: AlgorithmNodeCheckpointPayload,
) -> dict[str, object]:
    """Return split-layout runtime payload for one node."""
    return {
        "node_id": node_payload.node_id,
        "unopened_branches": node_payload.unopened_branches,
        "linked_children": node_payload.linked_children,
        "evaluation": node_payload.evaluation,
        "exploration_index": node_payload.exploration_index,
    }
