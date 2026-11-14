"""Streaming bytes and records for the typed materializer, without runtime IO."""

import json
from compression.zstd import DecompressionParameter, ZstdDecompressor
from hashlib import sha256
from typing import TYPE_CHECKING, cast

from .sharded_errors import ShardedCheckpointManifestError
from .sharded_io import _resolve_manifest_shard_path

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .sharded_read_limits import ShardedCheckpointReadLimits
    from .sharded_types import ShardedCheckpointShardRef

_CHUNK_BYTES = 64 * 1024


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ShardedCheckpointManifestError.invalid_input("duplicate_JSON_key")
        result[key] = value
    return result


class _ShardReader:
    """Own one cumulative budget; never keep a whole raw JSON mirror."""

    def __init__(self, root: Path, limits: ShardedCheckpointReadLimits | None) -> None:
        """Initialize one operation's actual byte counters."""
        self.root = root
        self.limits = limits
        self.compressed_bytes = 0
        self.uncompressed_bytes = 0
        self.peak_record_bytes = 0

    def check(self) -> None:
        """Observe the same deadline and cancellation throughout the read."""
        if self.limits is not None:
            self.limits.check()

    def bound(self, actual: int, name: str) -> None:
        """Refuse an exceeded cooperative allocation or input bound."""
        self.check()
        if self.limits is not None and actual > getattr(self.limits, name):
            raise ShardedCheckpointManifestError.invalid_input(f"{name}_exceeded")

    def _read_size(self, used: int, name: str) -> int:
        if self.limits is None:
            return _CHUNK_BYTES
        return min(_CHUNK_BYTES, cast("int", getattr(self.limits, name)) - used + 1)

    def _chunks(
        self, path: str, *, compressed: bool, shard: ShardedCheckpointShardRef | None
    ) -> Iterator[bytes]:
        self.check()
        before_compressed = self.compressed_bytes
        before_uncompressed = self.uncompressed_bytes
        digest = sha256()
        decoder: ZstdDecompressor | None = None
        finished_frame = False
        with _resolve_manifest_shard_path(self.root, path).open("rb") as stream:
            while True:
                self.check()
                raw = stream.read(
                    self._read_size(self.compressed_bytes, "max_compressed_bytes")
                )
                self.compressed_bytes += len(raw)
                self.bound(self.compressed_bytes, "max_compressed_bytes")
                digest.update(raw)
                if not raw:
                    break
                if not compressed:
                    self.uncompressed_bytes += len(raw)
                    self.bound(self.uncompressed_bytes, "max_uncompressed_bytes")
                    yield raw
                    continue
                while raw or (decoder is not None and not decoder.needs_input):
                    self.check()
                    if decoder is None:
                        options: dict[int, int] | None = None
                        if self.limits is not None:
                            # Bound the native window as well as returned output.
                            window_log = max(
                                10,
                                min(
                                    31,
                                    (
                                        self.limits.max_uncompressed_bytes - 1
                                    ).bit_length(),
                                ),
                            )
                            options = {
                                DecompressionParameter.window_log_max: window_log
                            }
                        decoder = ZstdDecompressor(options=options)
                        finished_frame = False
                    data = decoder.decompress(
                        raw,
                        max_length=self._read_size(
                            self.uncompressed_bytes, "max_uncompressed_bytes"
                        ),
                    )
                    self.uncompressed_bytes += len(data)
                    self.bound(self.uncompressed_bytes, "max_uncompressed_bytes")
                    raw = b""
                    if decoder.eof:
                        raw = decoder.unused_data
                        decoder = None
                        finished_frame = True
                    if data:
                        yield data
        if compressed and not finished_frame:
            raise ShardedCheckpointManifestError.invalid_input("truncated_zstd_frame")
        if shard is not None:
            for declared, actual in (
                (shard.compressed_bytes, self.compressed_bytes - before_compressed),
                (
                    shard.uncompressed_bytes,
                    self.uncompressed_bytes - before_uncompressed,
                ),
            ):
                if declared is not None and declared != actual:
                    raise ShardedCheckpointManifestError.invalid_input(
                        "shard_byte_declaration_mismatch"
                    )
            if shard.sha256 is not None and shard.sha256 != digest.hexdigest():
                raise ShardedCheckpointManifestError.invalid_input(
                    "shard_digest_mismatch"
                )
        self.check()

    def _append(self, record: bytearray, part: bytes) -> None:
        size = len(record) + len(part)
        self.bound(size, "max_record_bytes")
        self.peak_record_bytes = max(self.peak_record_bytes, size)
        record.extend(part)

    def _mapping(self, raw: bytearray) -> dict[str, object]:
        self.check()
        value = json.loads(raw, object_pairs_hook=_unique_object)
        self.check()
        if not isinstance(value, dict):
            raise ShardedCheckpointManifestError.shard_payload_not_mapping()
        return cast("dict[str, object]", value)

    def mapping(
        self, path: str, shard: ShardedCheckpointShardRef | None = None
    ) -> dict[str, object]:
        """Read a bounded singleton, including the manifest."""
        if shard is not None and shard.record_count != 1:
            raise ShardedCheckpointManifestError.invalid_input("singleton_record_count")
        raw = bytearray()
        for chunk in self._chunks(
            path,
            compressed=shard is not None and shard.encoding == "json_zst",
            shard=shard,
        ):
            self._append(raw, chunk)
        return self._mapping(raw)

    def records(self, shard: ShardedCheckpointShardRef) -> Iterator[dict[str, object]]:
        """Parse one bounded JSONL row at a time; verify actual record coverage."""
        pending = bytearray()
        count = 0
        for chunk in self._chunks(
            shard.path, compressed=shard.encoding == "jsonl_zst", shard=shard
        ):
            parts = chunk.split(b"\n")
            for position, part in enumerate(parts):
                self._append(pending, part)
                if position < len(parts) - 1:
                    if pending.strip():
                        count += 1
                        if count > shard.record_count:
                            raise ShardedCheckpointManifestError.invalid_input(
                                "shard_record_count_mismatch"
                            )
                        yield self._mapping(pending)
                    pending.clear()
        if pending.strip():
            count += 1
            if count > shard.record_count:
                raise ShardedCheckpointManifestError.invalid_input(
                    "shard_record_count_mismatch"
                )
            yield self._mapping(pending)
        if count != shard.record_count:
            raise ShardedCheckpointManifestError.invalid_input(
                "shard_record_count_mismatch"
            )
