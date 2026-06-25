"""Manifest primitives for future sharded runtime checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

from .io import checkpoint_payload_to_jsonable, write_checkpoint_json_payload
from .payloads import CHECKPOINT_FORMAT_VERSION, SearchRuntimeCheckpointPayload

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .io import CheckpointJsonEncoder

SHARDED_CHECKPOINT_FORMAT_VERSION = 1

type ShardedCheckpointShardKind = Literal[
    "metadata",
    "tree_nodes",
    "node_records",
    "state_payloads",
    "linked_children",
    "node_evaluations",
    "selector",
    "latest_expansions",
]
type ShardedCheckpointShardEncoding = Literal["json", "json_zst", "jsonl", "jsonl_zst"]


class ShardedCheckpointManifestError(ValueError):
    """Raised when a sharded checkpoint manifest is structurally invalid."""

    @classmethod
    def negative_shard_record_count(cls) -> ShardedCheckpointManifestError:
        """Return the error for a negative shard record count."""
        return cls("sharded checkpoint shard record_count must be non-negative.")

    @classmethod
    def negative_shard_compressed_bytes(cls) -> ShardedCheckpointManifestError:
        """Return the error for a negative shard compressed byte count."""
        return cls("sharded checkpoint shard compressed_bytes must be non-negative.")

    @classmethod
    def negative_shard_uncompressed_bytes(cls) -> ShardedCheckpointManifestError:
        """Return the error for a negative shard uncompressed byte count."""
        return cls("sharded checkpoint shard uncompressed_bytes must be non-negative.")

    @classmethod
    def negative_total_node_count(cls) -> ShardedCheckpointManifestError:
        """Return the error for a negative total node count."""
        return cls("sharded checkpoint total_node_count must be non-negative.")

    @classmethod
    def negative_total_branch_count(cls) -> ShardedCheckpointManifestError:
        """Return the error for a negative total branch count."""
        return cls("sharded checkpoint total_branch_count must be non-negative.")

    @classmethod
    def non_positive_node_count_per_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for an invalid node-count shard target."""
        return cls("sharded checkpoint node_count_per_shard must be positive.")

    @classmethod
    def unsupported_jsonl_encoding(cls) -> ShardedCheckpointManifestError:
        """Return the error for an unsupported JSONL shard encoding."""
        return cls("sharded checkpoint node records require jsonl or jsonl_zst encoding.")

    @classmethod
    def zstandard_required(cls) -> ShardedCheckpointManifestError:
        """Return the error for unavailable zstandard compression."""
        return cls("zstandard is required for jsonl_zst sharded checkpoints.")

    @classmethod
    def unsupported_checkpoint_format_version(cls) -> ShardedCheckpointManifestError:
        """Return the error for an unsupported base checkpoint format."""
        return cls(
            "unsupported monolithic checkpoint format version in sharded manifest."
        )

    @classmethod
    def unsupported_sharded_format_version(cls) -> ShardedCheckpointManifestError:
        """Return the error for an unsupported sharded checkpoint format."""
        return cls("unsupported sharded checkpoint format version.")

    @classmethod
    def manifest_not_mapping(cls) -> ShardedCheckpointManifestError:
        """Return the error for a non-mapping manifest payload."""
        return cls("sharded checkpoint manifest must be a mapping.")

    @classmethod
    def shards_not_list(cls) -> ShardedCheckpointManifestError:
        """Return the error for a non-list shard collection."""
        return cls("sharded checkpoint shards must be a list.")

    @classmethod
    def shard_not_mapping(cls) -> ShardedCheckpointManifestError:
        """Return the error for a non-mapping shard reference."""
        return cls("sharded checkpoint shard references must be mappings.")

    @classmethod
    def empty_path(cls) -> ShardedCheckpointManifestError:
        """Return the error for an empty shard path."""
        return cls("sharded checkpoint shard path is empty.")

    @classmethod
    def non_posix_path(cls) -> ShardedCheckpointManifestError:
        """Return the error for a shard path using non-POSIX separators."""
        return cls("sharded checkpoint shard paths must use POSIX separators.")

    @classmethod
    def unsafe_path(cls) -> ShardedCheckpointManifestError:
        """Return the error for an escaping or absolute shard path."""
        return cls(
            "sharded checkpoint shard paths must be relative and stay inside the checkpoint directory."
        )

    @classmethod
    def unnormalized_path(cls) -> ShardedCheckpointManifestError:
        """Return the error for a non-normal shard path."""
        return cls(
            "sharded checkpoint shard paths must be normalized relative paths."
        )


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
    return ShardedCheckpointManifest(
        checkpoint_format_version=cast(
            "int",
            data.get("checkpoint_format_version", CHECKPOINT_FORMAT_VERSION),
        ),
        sharded_checkpoint_format_version=cast(
            "int",
            data.get(
                "sharded_checkpoint_format_version",
                SHARDED_CHECKPOINT_FORMAT_VERSION,
            ),
        ),
        total_node_count=cast("int", data.get("total_node_count", 0)),
        total_branch_count=cast("int | None", data.get("total_branch_count")),
        generation=cast("int | None", data.get("generation")),
        node_count_per_shard=cast("int | None", data.get("node_count_per_shard")),
        encoder=cast("CheckpointJsonEncoder | None", data.get("encoder")),
        notes=cast("str | None", data.get("notes")),
        shards=tuple(
            _sharded_checkpoint_shard_ref_from_jsonable(raw_shard)
            for raw_shard in raw_shards
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


def write_sharded_search_checkpoint(
    payload: SearchRuntimeCheckpointPayload,
    output_dir: str | Path,
    *,
    node_count_per_shard: int = 100_000,
    encoding: ShardedCheckpointShardEncoding = "jsonl_zst",
    generation: int | None = None,
    encoder: CheckpointJsonEncoder | None = None,
) -> ShardedCheckpointManifest:
    """Write a generic first-stage sharded runtime checkpoint."""
    if node_count_per_shard <= 0:
        raise ShardedCheckpointManifestError.non_positive_node_count_per_shard()
    if encoding not in {"jsonl", "jsonl_zst"}:
        raise ShardedCheckpointManifestError.unsupported_jsonl_encoding()
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
        notes="C2b first-stage node-record shard layout.",
    )
    write_sharded_checkpoint_manifest(manifest, resolved_output_dir / "manifest.json")
    return manifest


def load_sharded_search_checkpoint(*_args: object, **_kwargs: object) -> None:
    """Raise until the generic sharded runtime checkpoint reader exists."""
    raise NotImplementedError(
        "Generic sharded runtime checkpoint loading is planned but not implemented."
    )


def _checkpoint_branch_count(payload: SearchRuntimeCheckpointPayload) -> int:
    """Return the number of linked child edges in one checkpoint payload."""
    return sum(len(node_payload.linked_children) for node_payload in payload.tree.nodes)


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


def _write_node_record_shards(
    node_payloads: Iterable[object],
    *,
    output_dir: Path,
    node_count_per_shard: int,
    encoding: ShardedCheckpointShardEncoding,
    encoder: CheckpointJsonEncoder,
) -> list[ShardedCheckpointShardRef]:
    """Write node-record JSONL shards and return manifest refs."""
    shards: list[ShardedCheckpointShardRef] = []
    current_records: list[object] = []
    shard_index = 0
    for node_payload in node_payloads:
        current_records.append(node_payload)
        if len(current_records) == node_count_per_shard:
            shards.append(
                _write_node_record_shard(
                    current_records,
                    output_dir=output_dir,
                    shard_index=shard_index,
                    encoding=encoding,
                    encoder=encoder,
                )
            )
            current_records = []
            shard_index += 1
    if current_records or not shards:
        shards.append(
            _write_node_record_shard(
                current_records,
                output_dir=output_dir,
                shard_index=shard_index,
                encoding=encoding,
                encoder=encoder,
            )
        )
    return shards


def _write_node_record_shard(
    records: list[object],
    *,
    output_dir: Path,
    shard_index: int,
    encoding: ShardedCheckpointShardEncoding,
    encoder: CheckpointJsonEncoder,
) -> ShardedCheckpointShardRef:
    suffix = "jsonl.zst" if encoding == "jsonl_zst" else "jsonl"
    relative_path = f"node_records/nodes_{shard_index:06d}.{suffix}"
    encoded_records = _encode_jsonl_records(records, encoder=encoder)
    compressed_records = _compress_jsonl_bytes(encoded_records, encoding)
    output_path = output_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(compressed_records)
    return ShardedCheckpointShardRef(
        kind="node_records",
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
    return zstandard_module.ZstdCompressor().compress(jsonl_bytes)


def _load_optional_module(module_name: str) -> object | None:
    """Import one optional module when available."""
    try:
        return import_module(module_name)
    except ImportError:
        return None


def _sharded_checkpoint_shard_ref_from_jsonable(
    payload: object,
) -> ShardedCheckpointShardRef:
    if not isinstance(payload, dict):
        raise ShardedCheckpointManifestError.shard_not_mapping()
    data = cast("dict[str, object]", payload)
    return ShardedCheckpointShardRef(
        kind=cast("ShardedCheckpointShardKind", data["kind"]),
        path=cast("str", data["path"]),
        record_count=cast("int", data["record_count"]),
        encoding=cast("ShardedCheckpointShardEncoding", data["encoding"]),
        compressed_bytes=cast("int | None", data.get("compressed_bytes")),
        uncompressed_bytes=cast("int | None", data.get("uncompressed_bytes")),
        sha256=cast("str | None", data.get("sha256")),
    )


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


__all__ = [
    "SHARDED_CHECKPOINT_FORMAT_VERSION",
    "ShardedCheckpointManifest",
    "ShardedCheckpointManifestError",
    "ShardedCheckpointShardEncoding",
    "ShardedCheckpointShardKind",
    "ShardedCheckpointShardRef",
    "load_sharded_search_checkpoint",
    "read_sharded_checkpoint_manifest",
    "sharded_checkpoint_manifest_from_jsonable",
    "sharded_checkpoint_manifest_to_jsonable",
    "write_sharded_checkpoint_manifest",
    "write_sharded_search_checkpoint",
]
