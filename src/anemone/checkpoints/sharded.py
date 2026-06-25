"""Manifest primitives for future sharded runtime checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

from .io import (
    checkpoint_payload_to_jsonable,
    load_checkpoint_json_payload,
    write_checkpoint_json_payload,
)
from .payloads import (
    CHECKPOINT_FORMAT_VERSION,
    AlgorithmNodeCheckpointPayload,
    AnchorCheckpointStatePayload,
    BackupRuntimeCheckpointPayload,
    BranchFrontierCheckpointPayload,
    BranchOrderingCheckpointPayload,
    CheckpointAtomPayload,
    CheckpointNodeStatePayload,
    DecisionOrderingCheckpointPayload,
    DeltaCheckpointStatePayload,
    ExplorationIndexCheckpointPayload,
    LinkedChildCheckpointPayload,
    LinooCandidateCheckpointPayload,
    LinooCandidatesByDepthCheckpointPayload,
    LinooDepthStatsCheckpointPayload,
    LinooNodeStateCheckpointPayload,
    LinooSelectorCheckpointPayload,
    NodeEvaluationCheckpointPayload,
    PrincipalVariationCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    SerializedOverEventPayload,
    SerializedValuePayload,
    TreeCheckpointPayload,
    TreeExpansionCheckpointPayload,
    TreeExpansionsCheckpointPayload,
)

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
    def unsupported_shard_kind(cls) -> ShardedCheckpointManifestError:
        """Return the error for an unsupported shard kind."""
        return cls("sharded checkpoint manifest contains an unsupported shard kind.")

    @classmethod
    def unsupported_shard_encoding(cls) -> ShardedCheckpointManifestError:
        """Return the error for an unsupported shard encoding."""
        return cls("sharded checkpoint manifest contains an unsupported shard encoding.")

    @classmethod
    def missing_metadata_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for a missing metadata shard."""
        return cls("sharded checkpoint manifest must contain exactly one metadata shard.")

    @classmethod
    def duplicate_metadata_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for multiple metadata shards."""
        return cls("sharded checkpoint manifest contains multiple metadata shards.")

    @classmethod
    def missing_node_records_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for missing node-record shards."""
        return cls("sharded checkpoint manifest must contain node-record shards.")

    @classmethod
    def duplicate_selector_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for multiple selector shards."""
        return cls("sharded checkpoint manifest contains multiple selector shards.")

    @classmethod
    def duplicate_latest_expansions_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for multiple latest-expansions shards."""
        return cls(
            "sharded checkpoint manifest contains multiple latest-expansions shards."
        )

    @classmethod
    def node_count_mismatch(cls) -> ShardedCheckpointManifestError:
        """Return the error for mismatched node counts."""
        return cls("sharded checkpoint node-record count does not match the manifest.")

    @classmethod
    def branch_count_mismatch(cls) -> ShardedCheckpointManifestError:
        """Return the error for mismatched branch counts."""
        return cls("sharded checkpoint branch count does not match the manifest.")

    @classmethod
    def shard_path_outside_root(cls) -> ShardedCheckpointManifestError:
        """Return the error for a resolved shard path outside the checkpoint root."""
        return cls("sharded checkpoint shard path resolves outside the checkpoint root.")

    @classmethod
    def shard_payload_not_mapping(cls) -> ShardedCheckpointManifestError:
        """Return the error for a shard payload that is not a JSON object."""
        return cls("sharded checkpoint shard payload must be a mapping.")

    @classmethod
    def node_record_not_mapping(cls) -> ShardedCheckpointManifestError:
        """Return the error for a node-record line that is not a JSON object."""
        return cls("sharded checkpoint node records must be JSON objects.")

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


def load_sharded_search_checkpoint(
    checkpoint_dir: str | Path,
) -> SearchRuntimeCheckpointPayload:
    """Load a C2b sharded runtime checkpoint as a typed checkpoint payload.

    C2c intentionally reconstructs the monolithic typed payload first. Later
    streaming restore work can replace this internal assembly path without
    changing the C2b manifest contract.
    """
    resolved_checkpoint_dir = Path(checkpoint_dir)
    manifest = read_sharded_checkpoint_manifest(
        resolved_checkpoint_dir / "manifest.json"
    )
    grouped_shards = _group_manifest_shards(manifest)

    metadata = _load_mapping_json_shard(
        resolved_checkpoint_dir,
        _only_manifest_shard(grouped_shards, "metadata"),
    )
    node_records = [
        record
        for shard in _required_manifest_shards(grouped_shards, "node_records")
        for record in _read_jsonl_shard_records(resolved_checkpoint_dir, shard)
    ]
    if len(node_records) != manifest.total_node_count:
        raise ShardedCheckpointManifestError.node_count_mismatch()
    if manifest.total_branch_count is not None:
        branch_count = sum(
            len(cast("list[object]", record.get("linked_children", [])))
            for record in node_records
        )
        if branch_count != manifest.total_branch_count:
            raise ShardedCheckpointManifestError.branch_count_mismatch()

    selector_state = _optional_mapping_json_shard(
        resolved_checkpoint_dir,
        grouped_shards,
        "selector",
    )
    latest_tree_expansions = _optional_mapping_json_shard(
        resolved_checkpoint_dir,
        grouped_shards,
        "latest_expansions",
    )

    return SearchRuntimeCheckpointPayload(
        format_version=cast("int", metadata["format_version"]),
        evaluator_version=cast("int", metadata["evaluator_version"]),
        rng_state=metadata.get("rng_state"),
        tree=TreeCheckpointPayload(
            root_node_id=cast("int", metadata["root_node_id"]),
            nodes=[
                _algorithm_node_payload_from_mapping(record)
                for record in node_records
            ],
        ),
        latest_tree_expansions=(
            None
            if latest_tree_expansions is None
            else _tree_expansions_payload_from_mapping(latest_tree_expansions)
        ),
        selector_state=(
            None
            if selector_state is None
            else _selector_payload_from_mapping(selector_state)
        ),
    )


def _group_manifest_shards(
    manifest: ShardedCheckpointManifest,
) -> dict[ShardedCheckpointShardKind, list[ShardedCheckpointShardRef]]:
    """Validate and group C2b manifest shards by kind."""
    grouped: dict[ShardedCheckpointShardKind, list[ShardedCheckpointShardRef]] = {}
    for shard in manifest.shards:
        if shard.kind not in {
            "metadata",
            "node_records",
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
    _required_manifest_shards(grouped, "node_records")
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
    if shard.encoding == "jsonl":
        shard_bytes = _resolve_manifest_shard_path(
            checkpoint_dir,
            shard.path,
        ).read_bytes()
    elif shard.encoding == "jsonl_zst":
        zstandard_module = _load_optional_module("zstandard")
        if zstandard_module is None:
            raise ShardedCheckpointManifestError.zstandard_required()
        shard_bytes = zstandard_module.ZstdDecompressor().decompress(
            _resolve_manifest_shard_path(checkpoint_dir, shard.path).read_bytes()
        )
    else:
        raise ShardedCheckpointManifestError.unsupported_jsonl_encoding()

    records: list[dict[str, object]] = []
    for line in shard_bytes.decode("utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ShardedCheckpointManifestError.node_record_not_mapping()
        records.append(cast("dict[str, object]", record))
    return records


def _resolve_manifest_shard_path(checkpoint_dir: Path, relative_path: str) -> Path:
    """Resolve a manifest shard path while keeping it inside the checkpoint root."""
    _validate_relative_manifest_path(relative_path)
    resolved_root = checkpoint_dir.resolve()
    resolved_path = (resolved_root / relative_path).resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ShardedCheckpointManifestError.shard_path_outside_root()
    return resolved_path


def _algorithm_node_payload_from_mapping(
    payload: dict[str, object],
) -> AlgorithmNodeCheckpointPayload:
    """Build one typed node payload from a C2b JSON record."""
    return AlgorithmNodeCheckpointPayload(
        node_id=cast("int", payload["node_id"]),
        parent_node_id=cast("int | None", payload["parent_node_id"]),
        branch_from_parent=cast(
            "CheckpointAtomPayload | None",
            payload["branch_from_parent"],
        ),
        depth=cast("int", payload["depth"]),
        state_payload=_state_payload_from_mapping(
            cast("dict[str, object]", payload["state_payload"])
        ),
        generated_all_branches=cast("bool", payload["generated_all_branches"]),
        unopened_branches=cast(
            "list[CheckpointAtomPayload]",
            payload.get("unopened_branches", []),
        ),
        linked_children=[
            _linked_child_payload_from_mapping(cast("dict[str, object]", child))
            for child in cast("list[object]", payload.get("linked_children", []))
        ],
        evaluation=(
            None
            if payload.get("evaluation") is None
            else _node_evaluation_payload_from_mapping(
                cast("dict[str, object]", payload["evaluation"])
            )
        ),
        exploration_index=(
            None
            if payload.get("exploration_index") is None
            else _exploration_index_payload_from_mapping(
                cast("dict[str, object]", payload["exploration_index"])
            )
        ),
    )


def _state_payload_from_mapping(
    payload: dict[str, object],
) -> CheckpointNodeStatePayload:
    """Build one typed checkpoint state payload from a JSON object."""
    if "anchor_ref" in payload:
        return AnchorCheckpointStatePayload(
            anchor_ref=payload["anchor_ref"],
            state_summary=payload.get("state_summary"),
        )
    return DeltaCheckpointStatePayload(
        state_parent_node_id=cast("int", payload["state_parent_node_id"]),
        state_parent_branch=cast(
            "CheckpointAtomPayload | None",
            payload["state_parent_branch"],
        ),
        delta_ref=payload["delta_ref"],
        state_summary=payload.get("state_summary"),
    )


def _linked_child_payload_from_mapping(
    payload: dict[str, object],
) -> LinkedChildCheckpointPayload:
    """Build one typed linked-child payload from a JSON object."""
    return LinkedChildCheckpointPayload(
        branch_key=cast("CheckpointAtomPayload", payload["branch_key"]),
        child_node_id=cast("int", payload["child_node_id"]),
    )


def _node_evaluation_payload_from_mapping(
    payload: dict[str, object],
) -> NodeEvaluationCheckpointPayload:
    """Build one typed node-evaluation payload from a JSON object."""
    return NodeEvaluationCheckpointPayload(
        direct_value=_optional_serialized_value(payload.get("direct_value")),
        direct_evaluation_version=cast(
            "int | None",
            payload.get("direct_evaluation_version"),
        ),
        backed_up_value=_optional_serialized_value(payload.get("backed_up_value")),
        decision_ordering=(
            None
            if payload.get("decision_ordering") is None
            else _decision_ordering_payload_from_mapping(
                cast("dict[str, object]", payload["decision_ordering"])
            )
        ),
        principal_variation=(
            None
            if payload.get("principal_variation") is None
            else _principal_variation_payload_from_mapping(
                cast("dict[str, object]", payload["principal_variation"])
            )
        ),
        branch_frontier=(
            None
            if payload.get("branch_frontier") is None
            else BranchFrontierCheckpointPayload(
                frontier_branches=cast(
                    "list[CheckpointAtomPayload]",
                    cast("dict[str, object]", payload["branch_frontier"]).get(
                        "frontier_branches",
                        [],
                    ),
                )
            )
        ),
        backup_runtime=(
            None
            if payload.get("backup_runtime") is None
            else _backup_runtime_payload_from_mapping(
                cast("dict[str, object]", payload["backup_runtime"])
            )
        ),
    )


def _optional_serialized_value(payload: object) -> SerializedValuePayload | None:
    """Build an optional typed serialized value from a JSON object."""
    if payload is None:
        return None
    return _serialized_value_payload_from_mapping(cast("dict[str, object]", payload))


def _serialized_value_payload_from_mapping(
    payload: dict[str, object],
) -> SerializedValuePayload:
    """Build one typed serialized value payload from a JSON object."""
    return SerializedValuePayload(
        score=cast("float", payload["score"]),
        certainty=cast("str", payload["certainty"]),
        over_event=(
            None
            if payload.get("over_event") is None
            else SerializedOverEventPayload(
                outcome=cast(
                    "str",
                    cast("dict[str, object]", payload["over_event"])["outcome"],
                ),
                termination=cast(
                    "CheckpointAtomPayload",
                    cast("dict[str, object]", payload["over_event"]).get(
                        "termination"
                    ),
                ),
                winner=cast(
                    "CheckpointAtomPayload",
                    cast("dict[str, object]", payload["over_event"]).get("winner"),
                ),
            )
        ),
        line=cast("list[CheckpointAtomPayload] | None", payload.get("line")),
    )


def _decision_ordering_payload_from_mapping(
    payload: dict[str, object],
) -> DecisionOrderingCheckpointPayload:
    """Build one typed decision-ordering payload from a JSON object."""
    return DecisionOrderingCheckpointPayload(
        branch_ordering=[
            BranchOrderingCheckpointPayload(
                branch_key=cast("CheckpointAtomPayload", ordering["branch_key"]),
                primary_score=cast("float", ordering["primary_score"]),
                tactical_tiebreak=cast("int", ordering["tactical_tiebreak"]),
                stable_tiebreak_id=cast("int", ordering["stable_tiebreak_id"]),
            )
            for ordering in cast(
                "list[dict[str, object]]",
                payload.get("branch_ordering", []),
            )
        ]
    )


def _principal_variation_payload_from_mapping(
    payload: dict[str, object],
) -> PrincipalVariationCheckpointPayload:
    """Build one typed principal-variation payload from a JSON object."""
    return PrincipalVariationCheckpointPayload(
        best_branch_sequence=cast(
            "list[CheckpointAtomPayload]",
            payload.get("best_branch_sequence", []),
        ),
        pv_version=cast("int", payload.get("pv_version", 0)),
        cached_best_child_version=cast(
            "int | None",
            payload.get("cached_best_child_version"),
        ),
    )


def _backup_runtime_payload_from_mapping(
    payload: dict[str, object],
) -> BackupRuntimeCheckpointPayload:
    """Build one typed backup-runtime payload from a JSON object."""
    return BackupRuntimeCheckpointPayload(
        best_branch=cast("CheckpointAtomPayload | None", payload.get("best_branch")),
        second_best_branch=cast(
            "CheckpointAtomPayload | None",
            payload.get("second_best_branch"),
        ),
        exact_child_count=cast("int", payload.get("exact_child_count", 0)),
        selected_child_pv_version=cast(
            "int | None",
            payload.get("selected_child_pv_version"),
        ),
        is_initialized=cast("bool", payload.get("is_initialized", False)),
    )


def _exploration_index_payload_from_mapping(
    payload: dict[str, object],
) -> ExplorationIndexCheckpointPayload:
    """Build one typed exploration-index payload from a JSON object."""
    return ExplorationIndexCheckpointPayload(
        kind=cast("str | None", payload.get("kind")),
        payload=payload.get("payload"),
    )


def _tree_expansions_payload_from_mapping(
    payload: dict[str, object],
) -> TreeExpansionsCheckpointPayload:
    """Build typed latest-expansions payload from a JSON object."""
    return TreeExpansionsCheckpointPayload(
        expansions_with_node_creation=[
            _tree_expansion_payload_from_mapping(cast("dict[str, object]", expansion))
            for expansion in cast(
                "list[object]",
                payload.get("expansions_with_node_creation", []),
            )
        ],
        expansions_without_node_creation=[
            _tree_expansion_payload_from_mapping(cast("dict[str, object]", expansion))
            for expansion in cast(
                "list[object]",
                payload.get("expansions_without_node_creation", []),
            )
        ],
    )


def _tree_expansion_payload_from_mapping(
    payload: dict[str, object],
) -> TreeExpansionCheckpointPayload:
    """Build one typed tree-expansion payload from a JSON object."""
    return TreeExpansionCheckpointPayload(
        child_node_id=cast("int", payload["child_node_id"]),
        parent_node_id=cast("int | None", payload["parent_node_id"]),
        branch_key=cast("CheckpointAtomPayload | None", payload["branch_key"]),
        creation_child_node=cast("bool", payload["creation_child_node"]),
    )


def _selector_payload_from_mapping(
    payload: dict[str, object],
) -> LinooSelectorCheckpointPayload:
    """Build the generic selector payload currently supported by Anemone."""
    return LinooSelectorCheckpointPayload(
        type=cast("Literal['linoo']", payload.get("type", "linoo")),
        version=cast("int", payload.get("version", 1)),
        depth_stats=[
            LinooDepthStatsCheckpointPayload(
                depth=cast("int", depth_stats["depth"]),
                total_nodes=cast("int", depth_stats["total_nodes"]),
                opened_count=cast("int", depth_stats["opened_count"]),
                frontier_count=cast("int", depth_stats["frontier_count"]),
                terminal_count=cast("int", depth_stats["terminal_count"]),
                exact_count=cast("int", depth_stats["exact_count"]),
                uncached_terminal_candidates=cast(
                    "int",
                    depth_stats["uncached_terminal_candidates"],
                ),
                non_openable_count=cast("int", depth_stats["non_openable_count"]),
            )
            for depth_stats in cast(
                "list[dict[str, object]]",
                payload.get("depth_stats", []),
            )
        ],
        node_states=[
            LinooNodeStateCheckpointPayload(
                node_id=cast("int", node_state["node_id"]),
                depth=cast("int", node_state["depth"]),
                status=cast("str", node_state["status"]),
            )
            for node_state in cast(
                "list[dict[str, object]]",
                payload.get("node_states", []),
            )
        ],
        candidates_by_depth=[
            _linoo_candidates_by_depth_payload_from_mapping(candidates_by_depth)
            for candidates_by_depth in cast(
                "list[dict[str, object]]",
                payload.get("candidates_by_depth", []),
            )
        ],
        last_selected_node_id=cast("int | None", payload.get("last_selected_node_id")),
    )


def _linoo_candidates_by_depth_payload_from_mapping(
    payload: dict[str, object],
) -> LinooCandidatesByDepthCheckpointPayload:
    """Build one typed Linoo candidates-by-depth payload from a JSON object."""
    return LinooCandidatesByDepthCheckpointPayload(
        depth=cast("int", payload["depth"]),
        candidates=[
            LinooCandidateCheckpointPayload(
                node_id=cast("int", candidate["node_id"]),
                depth=cast("int", candidate["depth"]),
                priority=cast("float", candidate["priority"]),
                version=cast("int", candidate["version"]),
            )
            for candidate in cast(
                "list[dict[str, object]]",
                payload.get("candidates", []),
            )
        ],
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
