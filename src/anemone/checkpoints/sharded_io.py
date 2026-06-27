"""Low-level shard IO helpers for sharded checkpoints."""

from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnusedFunction=false
import io
import json
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from .io import (
    checkpoint_payload_to_jsonable,
    load_checkpoint_json_payload,
    write_checkpoint_json_payload,
)
from .sharded_errors import ShardedCheckpointManifestError
from .sharded_types import (
    ShardedCheckpointManifest,
    ShardedCheckpointShardEncoding,
    ShardedCheckpointShardKind,
    ShardedCheckpointShardRef,
    _validate_relative_manifest_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from .io import CheckpointJsonEncoder


def _group_manifest_shards(
    manifest: ShardedCheckpointManifest,
) -> dict[ShardedCheckpointShardKind, list[ShardedCheckpointShardRef]]:
    """Validate and group manifest shards by kind."""
    grouped: dict[ShardedCheckpointShardKind, list[ShardedCheckpointShardRef]] = {}
    for shard in manifest.shards:
        if shard.kind not in {
            "metadata",
            "node_records",
            "node_shells",
            "state_payloads",
            "node_runtime",
            "selector",
            "latest_expansions",
        }:
            raise ShardedCheckpointManifestError.unsupported_shard_kind()
        if shard.kind in {"metadata", "selector", "latest_expansions"}:
            if shard.encoding not in {"json", "json_zst"}:
                raise ShardedCheckpointManifestError.unsupported_shard_encoding()
        elif shard.encoding not in {"jsonl", "jsonl_zst"}:
            raise ShardedCheckpointManifestError.unsupported_jsonl_encoding()
        grouped.setdefault(shard.kind, []).append(shard)

    _only_manifest_shard(grouped, "metadata")
    if not grouped.get("node_records"):
        _required_manifest_shards(grouped, "node_shells")
        _required_manifest_shards(grouped, "state_payloads")
        _required_manifest_shards(grouped, "node_runtime")
    if len(grouped.get("selector", [])) > 1:
        raise ShardedCheckpointManifestError.duplicate_selector_shard()
    if len(grouped.get("latest_expansions", [])) > 1:
        raise ShardedCheckpointManifestError.duplicate_latest_expansions_shard()
    return grouped


def _only_manifest_shard(
    grouped_shards: dict[ShardedCheckpointShardKind, list[ShardedCheckpointShardRef]],
    kind: ShardedCheckpointShardKind,
) -> ShardedCheckpointShardRef:
    """Return the single required shard for a singleton kind."""
    shards = grouped_shards.get(kind, [])
    if not shards and kind == "metadata":
        raise ShardedCheckpointManifestError.missing_metadata_shard()
    if len(shards) > 1 and kind == "metadata":
        raise ShardedCheckpointManifestError.duplicate_metadata_shard()
    return shards[0]


def _required_manifest_shards(
    grouped_shards: dict[ShardedCheckpointShardKind, list[ShardedCheckpointShardRef]],
    kind: ShardedCheckpointShardKind,
) -> list[ShardedCheckpointShardRef]:
    """Return the required shard list for a repeated kind."""
    shards = grouped_shards.get(kind, [])
    if not shards and kind == "node_records":
        raise ShardedCheckpointManifestError.missing_node_records_shard()
    if not shards and kind == "node_shells":
        raise ShardedCheckpointManifestError.missing_node_shells_shard()
    if not shards and kind == "state_payloads":
        raise ShardedCheckpointManifestError.missing_state_payloads_shard()
    if not shards and kind == "node_runtime":
        raise ShardedCheckpointManifestError.missing_node_runtime_shard()
    return shards


def _optional_mapping_json_shard(
    checkpoint_dir: Path,
    grouped_shards: dict[ShardedCheckpointShardKind, list[ShardedCheckpointShardRef]],
    kind: ShardedCheckpointShardKind,
) -> dict[str, object] | None:
    """Load an optional singleton whole-JSON shard as a mapping."""
    shards = grouped_shards.get(kind, [])
    if not shards:
        return None
    return _load_mapping_json_shard(checkpoint_dir, shards[0])


def _load_mapping_json_shard(
    checkpoint_dir: Path,
    shard: ShardedCheckpointShardRef,
) -> dict[str, object]:
    """Load one whole-JSON shard and require an object payload."""
    payload, _stats = load_checkpoint_json_payload(
        _resolve_manifest_shard_path(checkpoint_dir, shard.path)
    )
    if not isinstance(payload, dict):
        raise ShardedCheckpointManifestError.shard_payload_not_mapping()
    return cast("dict[str, object]", payload)


def _read_jsonl_shard_records(
    checkpoint_dir: Path,
    shard: ShardedCheckpointShardRef,
) -> list[dict[str, object]]:
    """Read one JSONL or JSONL+zstd node-record shard."""
    return list(_iter_jsonl_shard_records(checkpoint_dir, shard))


def _iter_jsonl_shard_record_batches(
    checkpoint_dir: Path,
    shard: ShardedCheckpointShardRef,
    *,
    batch_size: int,
) -> Iterator[list[dict[str, object]]]:
    """Yield bounded JSON object batches from one JSONL shard."""
    if batch_size <= 0:
        raise ShardedCheckpointManifestError.non_positive_restore_batch_size()
    batch: list[dict[str, object]] = []
    for record in _iter_jsonl_shard_records(checkpoint_dir, shard):
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_jsonl_shard_records(
    checkpoint_dir: Path,
    shard: ShardedCheckpointShardRef,
) -> Iterator[dict[str, object]]:
    """Iterate JSON objects from one JSONL or JSONL+zstd shard."""
    if shard.encoding == "jsonl":
        with _resolve_manifest_shard_path(checkpoint_dir, shard.path).open(
            encoding="utf-8"
        ) as text_stream:
            yield from _iter_jsonl_records_from_lines(text_stream)
        return
    if shard.encoding == "jsonl_zst":
        zstandard_module = _load_optional_module("zstandard")
        if zstandard_module is None:
            raise ShardedCheckpointManifestError.zstandard_required()
        with (
            _resolve_manifest_shard_path(checkpoint_dir, shard.path).open(
                "rb"
            ) as compressed_stream,
            zstandard_module.ZstdDecompressor().stream_reader(
                compressed_stream
            ) as binary_stream,
        ):
            text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8")
            yield from _iter_jsonl_records_from_lines(text_stream)
        return
    raise ShardedCheckpointManifestError.unsupported_jsonl_encoding()


def _iter_jsonl_records_from_lines(
    lines: Iterable[str],
) -> Iterator[dict[str, object]]:
    """Yield JSON object records from text lines."""
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ShardedCheckpointManifestError.node_record_not_mapping()
        yield cast("dict[str, object]", record)


def _resolve_manifest_shard_path(checkpoint_dir: Path, relative_path: str) -> Path:
    """Resolve a manifest shard path while keeping it inside the checkpoint root."""
    _validate_relative_manifest_path(relative_path)
    resolved_root = checkpoint_dir.resolve()
    resolved_path = (resolved_root / relative_path).resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ShardedCheckpointManifestError.shard_path_outside_root()
    return resolved_path


def _write_json_shard(
    payload: object,
    *,
    output_dir: Path,
    relative_path: str,
    kind: ShardedCheckpointShardKind,
) -> ShardedCheckpointShardRef:
    """Write one compressed whole-JSON shard and return its manifest ref."""
    output_path = output_dir / relative_path
    write_stats = write_checkpoint_json_payload(payload, output_path)
    return ShardedCheckpointShardRef(
        kind=kind,
        path=relative_path,
        record_count=1,
        encoding="json_zst",
        compressed_bytes=write_stats.compressed_bytes,
        uncompressed_bytes=write_stats.uncompressed_bytes,
    )


def _write_jsonl_record_shards(
    records: Iterable[object],
    *,
    output_dir: Path,
    node_count_per_shard: int,
    encoding: ShardedCheckpointShardEncoding,
    encoder: CheckpointJsonEncoder,
    kind: ShardedCheckpointShardKind,
    directory_name: str,
    file_stem: str,
) -> list[ShardedCheckpointShardRef]:
    """Write JSONL record shards and return manifest refs."""
    shards: list[ShardedCheckpointShardRef] = []
    current_records: list[object] = []
    shard_index = 0
    for record in records:
        current_records.append(record)
        if len(current_records) == node_count_per_shard:
            shards.append(
                _write_jsonl_record_shard(
                    current_records,
                    output_dir=output_dir,
                    shard_index=shard_index,
                    encoding=encoding,
                    encoder=encoder,
                    kind=kind,
                    directory_name=directory_name,
                    file_stem=file_stem,
                )
            )
            current_records = []
            shard_index += 1
    if current_records or not shards:
        shards.append(
            _write_jsonl_record_shard(
                current_records,
                output_dir=output_dir,
                shard_index=shard_index,
                encoding=encoding,
                encoder=encoder,
                kind=kind,
                directory_name=directory_name,
                file_stem=file_stem,
            )
        )
    return shards


def _write_jsonl_record_shard(
    records: list[object],
    *,
    output_dir: Path,
    shard_index: int,
    encoding: ShardedCheckpointShardEncoding,
    encoder: CheckpointJsonEncoder,
    kind: ShardedCheckpointShardKind,
    directory_name: str,
    file_stem: str,
) -> ShardedCheckpointShardRef:
    suffix = "jsonl.zst" if encoding == "jsonl_zst" else "jsonl"
    relative_path = f"{directory_name}/{file_stem}_{shard_index:06d}.{suffix}"
    encoded_records = _encode_jsonl_records(records, encoder=encoder)
    compressed_records = _compress_jsonl_bytes(encoded_records, encoding)
    output_path = output_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(compressed_records)
    return ShardedCheckpointShardRef(
        kind=kind,
        path=relative_path,
        record_count=len(records),
        encoding=encoding,
        compressed_bytes=len(compressed_records),
        uncompressed_bytes=len(encoded_records),
    )


def _encode_jsonl_records(
    records: Iterable[object],
    *,
    encoder: CheckpointJsonEncoder,
) -> bytes:
    """Encode checkpoint records as newline-delimited JSON bytes."""
    if encoder == "orjson":
        orjson_module = import_module("orjson")
        return b"".join(
            orjson_module.dumps(checkpoint_payload_to_jsonable(record)) + b"\n"
            for record in records
        )
    return "".join(
        json.dumps(
            checkpoint_payload_to_jsonable(record),
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for record in records
    ).encode("utf-8")


def _compress_jsonl_bytes(
    jsonl_bytes: bytes,
    encoding: ShardedCheckpointShardEncoding,
) -> bytes:
    """Compress JSONL bytes according to the requested shard encoding."""
    if encoding == "jsonl":
        return jsonl_bytes
    if encoding != "jsonl_zst":
        raise ShardedCheckpointManifestError.unsupported_jsonl_encoding()
    zstandard_module = _load_optional_module("zstandard")
    if zstandard_module is None:
        raise ShardedCheckpointManifestError.zstandard_required()
    return cast("bytes", zstandard_module.ZstdCompressor().compress(jsonl_bytes))


def _load_optional_module(module_name: str) -> Any | None:
    """Import one optional module when available."""
    try:
        return import_module(module_name)
    except ImportError:
        return None
