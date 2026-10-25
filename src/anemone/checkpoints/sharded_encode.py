"""Write sharded checkpoint payloads to manifest and shard files."""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .build import _build_runtime_checkpoint_payload
from .build_context import (
    _CheckpointBuildContext,
    _CheckpointBuildMetrics,
    _maybe_reset_checkpoint_profile,
)
from .build_node_payloads import _iter_node_payloads
from .payloads import TreeCheckpointPayload
from .sharded_errors import ShardedCheckpointManifestError
from .sharded_io import (
    _write_json_shard,
    _write_jsonl_record_shard,
    _write_jsonl_record_shards,
)
from .sharded_manifest import write_sharded_checkpoint_manifest
from .sharded_types import (
    SHARDED_CHECKPOINT_FORMAT_VERSION,
    ShardedCheckpointLayout,
    ShardedCheckpointManifest,
    ShardedCheckpointShardEncoding,
    ShardedCheckpointShardRef,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import Any

    from anemone.tree_exploration import TreeExploration

    from ._protocols import IncrementalStateCheckpointCodec
    from .io import CheckpointJsonEncoder
    from .payloads import AlgorithmNodeCheckpointPayload, SearchRuntimeCheckpointPayload
    from .sharded_types import ShardedCheckpointShardKind


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
    return _write_sharded_checkpoint(
        payload.tree.nodes,
        metadata_payload=lambda: payload,
        output_dir=output_dir,
        node_count_per_shard=node_count_per_shard,
        encoding=encoding,
        layout=layout,
        generation=generation,
        encoder=encoder,
    )


def write_sharded_search_checkpoint_from_search(
    search: TreeExploration[Any],
    output_dir: str | Path,
    *,
    state_codec: IncrementalStateCheckpointCodec[Any],
    node_count_per_shard: int = 100_000,
    encoding: ShardedCheckpointShardEncoding = "jsonl_zst",
    layout: ShardedCheckpointLayout = "node_records",
    generation: int | None = None,
    encoder: CheckpointJsonEncoder | None = None,
) -> ShardedCheckpointManifest:
    """Stream a quiescent search into a new private checkpoint directory.

    Node payloads and serialization caches are bounded by the shard size and
    node/branch data. Selector and manifest metadata retain their existing size.
    The caller still owns validation, fsync and immutable publication.
    """
    metrics = _CheckpointBuildMetrics()
    context = _CheckpointBuildContext(metrics=metrics)
    _maybe_reset_checkpoint_profile(state_codec)
    return _write_sharded_checkpoint(
        _iter_node_payloads(
            search=search,
            state_codec=state_codec,
            metrics=metrics,
            context=context,
            cache_batch_size=node_count_per_shard,
        ),
        metadata_payload=lambda: _build_runtime_checkpoint_payload(
            search,
            tree_payload=TreeCheckpointPayload(root_node_id=search.tree.root_node.id),
            metrics=metrics,
            state_codec=state_codec,
        ),
        output_dir=output_dir,
        node_count_per_shard=node_count_per_shard,
        encoding=encoding,
        layout=layout,
        generation=generation,
        encoder=encoder,
    )


def _write_sharded_checkpoint(
    node_payloads: Iterable[AlgorithmNodeCheckpointPayload],
    *,
    metadata_payload: Callable[[], SearchRuntimeCheckpointPayload],
    output_dir: str | Path,
    node_count_per_shard: int,
    encoding: ShardedCheckpointShardEncoding,
    layout: ShardedCheckpointLayout,
    generation: int | None,
    encoder: CheckpointJsonEncoder | None,
) -> ShardedCheckpointManifest:
    """Use one shard/manifest writer for materialized and streaming construction."""
    if node_count_per_shard <= 0:
        raise ShardedCheckpointManifestError.non_positive_node_count_per_shard()
    if encoding not in {"jsonl", "jsonl_zst"}:
        raise ShardedCheckpointManifestError.unsupported_jsonl_encoding()
    if layout not in {"node_records", "split"}:
        raise ShardedCheckpointManifestError.unsupported_layout()
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if any(resolved_output_dir.iterdir()):
        raise _nonempty_checkpoint_directory(resolved_output_dir)
    resolved_encoder: CheckpointJsonEncoder = "stdlib" if encoder is None else encoder

    node_count = branch_count = 0

    def counted_nodes() -> Iterator[AlgorithmNodeCheckpointPayload]:
        nonlocal node_count, branch_count
        for node in node_payloads:
            node_count += 1
            branch_count += len(node.linked_children)
            yield node

    if layout == "split":
        shards = _write_split_node_shards(
            counted_nodes(),
            output_dir=resolved_output_dir,
            node_count_per_shard=node_count_per_shard,
            encoding=encoding,
            encoder=resolved_encoder,
        )
    else:
        shards = _write_node_record_shards(
            counted_nodes(),
            output_dir=resolved_output_dir,
            node_count_per_shard=node_count_per_shard,
            encoding=encoding,
            encoder=resolved_encoder,
        )
    payload = metadata_payload()
    metadata = {
        "format_version": payload.format_version,
        "evaluator_version": payload.evaluator_version,
        "rng_state": payload.rng_state,
        "rollout_rng_state": payload.rollout_rng_state,
        "root_node_id": payload.tree.root_node_id,
        "node_count": node_count,
        "branch_count": branch_count,
    }
    shards.insert(
        0,
        _write_json_shard(
            metadata,
            output_dir=resolved_output_dir,
            relative_path="metadata.json.zst",
            kind="metadata",
        ),
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
        total_node_count=node_count,
        total_branch_count=branch_count,
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


def _nonempty_checkpoint_directory(path: Path) -> FileExistsError:
    return FileExistsError(f"Checkpoint output directory must be empty: {path}")


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
    shells: list[ShardedCheckpointShardRef] = []
    states: list[ShardedCheckpointShardRef] = []
    runtimes: list[ShardedCheckpointShardRef] = []
    families = (
        ("node_shells", "nodes", _node_shell_record, shells),
        ("state_payloads", "state_payloads", _state_payload_record, states),
        ("node_runtime", "node_runtime", _node_runtime_record, runtimes),
    )
    nodes = iter(node_payloads)
    shard_index = 0
    while True:
        batch = tuple(islice(nodes, node_count_per_shard))
        if not batch and shard_index:
            break
        for kind, file_stem, record_builder, refs in families:
            refs.append(
                _write_jsonl_record_shard(
                    [record_builder(node) for node in batch],
                    output_dir=output_dir,
                    shard_index=shard_index,
                    encoding=encoding,
                    encoder=encoder,
                    kind=cast("ShardedCheckpointShardKind", kind),
                    directory_name=kind,
                    file_stem=file_stem,
                )
            )
        if not batch:
            break
        del batch
        shard_index += 1
    return shells + states + runtimes


def _node_shell_record(
    node_payload: AlgorithmNodeCheckpointPayload,
) -> dict[str, object]:
    """Return split-layout compact node metadata for one node."""
    return {
        "node_id": node_payload.node_id,
        "parent_node_id": node_payload.parent_node_id,
        "parent_node_order": node_payload.parent_node_order,
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
