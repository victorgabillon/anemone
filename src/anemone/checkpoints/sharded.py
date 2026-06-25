"""Manifest primitives for future sharded runtime checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

from .payloads import CHECKPOINT_FORMAT_VERSION

if TYPE_CHECKING:
    from .io import CheckpointJsonEncoder

SHARDED_CHECKPOINT_FORMAT_VERSION = 1

type ShardedCheckpointShardKind = Literal[
    "metadata",
    "tree_nodes",
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


def write_sharded_search_checkpoint(*_args: object, **_kwargs: object) -> None:
    """Raise until the generic sharded runtime checkpoint writer exists."""
    raise NotImplementedError(
        "Generic sharded runtime checkpoint writing is planned but not implemented."
    )


def load_sharded_search_checkpoint(*_args: object, **_kwargs: object) -> None:
    """Raise until the generic sharded runtime checkpoint reader exists."""
    raise NotImplementedError(
        "Generic sharded runtime checkpoint loading is planned but not implemented."
    )


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
