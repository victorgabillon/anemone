"""Shared JSON serialization and file I/O helpers for checkpoints."""

from __future__ import annotations

import gzip
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - exercised in environments without orjson.
    _orjson = None

try:
    import zstandard as _zstandard
except ImportError:  # pragma: no cover - exercised in environments without zstandard.
    _zstandard = None


type CheckpointFileFormat = Literal["json", "json_gz", "json_zst"]
type CheckpointJsonEncoder = Literal["orjson", "stdlib"]

_CHECKPOINT_SUFFIX_BY_FORMAT: dict[CheckpointFileFormat, str] = {
    "json": ".json",
    "json_gz": ".json.gz",
    "json_zst": ".json.zst",
}
_CHECKPOINT_CLI_NAME_BY_FORMAT: dict[CheckpointFileFormat, str] = {
    "json": "json",
    "json_gz": "json-gz",
    "json_zst": "json-zst",
}
_CHECKPOINT_FORMAT_BY_CLI_NAME = {
    cli_name: file_format
    for file_format, cli_name in _CHECKPOINT_CLI_NAME_BY_FORMAT.items()
}
_GENERATION_CHECKPOINT_NAME = re.compile(
    r"^generation_(?P<generation>\d+)\.json(?:\.(?P<compression>gz|zst))?$"
)
_GENERATION_FORMAT_PREFERENCE: dict[CheckpointFileFormat, int] = {
    "json": 0,
    "json_gz": 1,
    "json_zst": 2,
}

DEFAULT_CHECKPOINT_FILE_FORMAT: CheckpointFileFormat = (
    "json_zst" if _zstandard is not None else "json_gz"
)

_SCALAR_TYPES = (type(None), bool, int, float, str)
_DATACLASS_FIELD_NAMES_BY_TYPE: dict[type[object], tuple[str, ...]] = {}


@dataclass(frozen=True, slots=True)
class CheckpointWriteStats:
    """Compact serialization and file-write metrics for one checkpoint save."""

    output_path: Path
    file_format: CheckpointFileFormat
    encoder: CheckpointJsonEncoder
    uncompressed_bytes: int
    compressed_bytes: int
    jsonable_s: float | None
    json_encode_s: float
    compress_s: float | None
    write_s: float

    @property
    def compression_ratio(self) -> float | None:
        """Return the compressed-size ratio when available."""
        if self.uncompressed_bytes <= 0:
            return None
        return self.compressed_bytes / self.uncompressed_bytes


@dataclass(frozen=True, slots=True)
class CheckpointReadStats:
    """Compact file-read metrics for one checkpoint load."""

    input_path: Path
    file_format: CheckpointFileFormat
    compressed_bytes: int
    json_load_s: float


def checkpoint_payload_to_jsonable(value: object) -> object:
    """Recursively convert checkpoint dataclasses into JSON-ready containers."""
    if isinstance(value, _SCALAR_TYPES):
        return value
    if isinstance(value, list):
        return [checkpoint_payload_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [checkpoint_payload_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            checkpoint_payload_to_jsonable(key): checkpoint_payload_to_jsonable(item)
            for key, item in value.items()
        }
    field_names = _dataclass_field_names(type(value))
    if field_names and not isinstance(value, type):
        return {
            field_name: checkpoint_payload_to_jsonable(getattr(value, field_name))
            for field_name in field_names
        }
    return value


def default_checkpoint_file_suffix() -> str:
    """Return the preferred runtime-checkpoint file suffix for this environment."""
    return checkpoint_file_suffix(DEFAULT_CHECKPOINT_FILE_FORMAT)


def checkpoint_file_suffix(file_format: CheckpointFileFormat) -> str:
    """Return the filename suffix for one checkpoint file format."""
    return _CHECKPOINT_SUFFIX_BY_FORMAT[file_format]


def checkpoint_cli_name(file_format: CheckpointFileFormat) -> str:
    """Return the CLI-facing name for one checkpoint file format."""
    return _CHECKPOINT_CLI_NAME_BY_FORMAT[file_format]


def checkpoint_format_from_cli_name(name: str) -> CheckpointFileFormat:
    """Parse a CLI-facing checkpoint format name."""
    return _CHECKPOINT_FORMAT_BY_CLI_NAME[name]


def checkpoint_path_for_generation(
    directory: str | Path,
    generation: int,
    *,
    file_format: CheckpointFileFormat | None = None,
) -> Path:
    """Return one generation checkpoint path using the preferred suffix."""
    resolved_directory = Path(directory)
    resolved_format = (
        DEFAULT_CHECKPOINT_FILE_FORMAT if file_format is None else file_format
    )
    return (
        resolved_directory
        / f"generation_{generation:06d}{checkpoint_file_suffix(resolved_format)}"
    )


def checkpoint_output_path(
    path: str | Path,
    *,
    file_format: CheckpointFileFormat | None = None,
) -> Path:
    """Return one output path normalized to the requested checkpoint suffix."""
    resolved_path = Path(path)
    resolved_format = (
        DEFAULT_CHECKPOINT_FILE_FORMAT if file_format is None else file_format
    )
    name = resolved_path.name
    for known_suffix in (".json.zst", ".json.gz", ".json"):
        if name.endswith(known_suffix):
            base_name = name.removesuffix(known_suffix)
            return resolved_path.with_name(
                f"{base_name}{checkpoint_file_suffix(resolved_format)}"
            )
    return resolved_path.with_name(
        f"{name}{checkpoint_file_suffix(resolved_format)}"
    )


def checkpoint_file_format_from_path(path: str | Path) -> CheckpointFileFormat:
    """Infer the checkpoint file format from one path suffix."""
    name = Path(path).name
    if name.endswith(".json.zst"):
        return "json_zst"
    if name.endswith(".json.gz"):
        return "json_gz"
    if name.endswith(".json"):
        return "json"
    raise ValueError(f"Unsupported checkpoint file suffix: {path!s}")


def parse_generation_checkpoint_name(path: str | Path) -> int | None:
    """Parse generation checkpoint filenames across plain and compressed suffixes."""
    match = _GENERATION_CHECKPOINT_NAME.match(Path(path).name)
    if match is None:
        return None
    return int(match.group("generation"))


def resolve_latest_generation_checkpoint_path(directory: str | Path) -> Path:
    """Return the newest generation checkpoint across supported suffixes."""
    resolved_directory = Path(directory)
    candidates: list[tuple[int, int, Path]] = []
    for path in resolved_directory.iterdir():
        if not path.is_file():
            continue
        generation = parse_generation_checkpoint_name(path)
        if generation is None:
            continue
        file_format = checkpoint_file_format_from_path(path)
        candidates.append(
            (
                generation,
                _GENERATION_FORMAT_PREFERENCE[file_format],
                path,
            )
        )
    if not candidates:
        raise FileNotFoundError(
            f"No runtime checkpoint found in {resolved_directory!s}."
        )
    _generation, _format_preference, latest_path = max(candidates)
    return latest_path


def write_checkpoint_json_payload(
    payload: object,
    output_path: str | Path,
) -> CheckpointWriteStats:
    """Encode one checkpoint payload to JSON, compress if needed, and write once."""
    resolved_output = Path(output_path)
    resolved_format = checkpoint_file_format_from_path(resolved_output)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    encoded_payload = _encode_checkpoint_payload(payload)
    compressed_payload, compress_s = _compress_checkpoint_bytes(
        encoded_payload.json_bytes,
        resolved_format,
    )
    write_started_at = perf_counter()
    resolved_output.write_bytes(compressed_payload)
    write_s = perf_counter() - write_started_at
    return CheckpointWriteStats(
        output_path=resolved_output,
        file_format=resolved_format,
        encoder=encoded_payload.encoder,
        uncompressed_bytes=len(encoded_payload.json_bytes),
        compressed_bytes=len(compressed_payload),
        jsonable_s=encoded_payload.jsonable_s,
        json_encode_s=encoded_payload.json_encode_s,
        compress_s=compress_s,
        write_s=write_s,
    )


def load_checkpoint_json_payload(
    input_path: str | Path,
) -> tuple[object, CheckpointReadStats]:
    """Load one JSON checkpoint payload across supported plain/compressed formats."""
    resolved_input = Path(input_path)
    resolved_format = checkpoint_file_format_from_path(resolved_input)
    compressed_bytes = resolved_input.stat().st_size
    load_started_at = perf_counter()
    if resolved_format == "json":
        with resolved_input.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    elif resolved_format == "json_gz":
        with gzip.open(resolved_input, mode="rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        compressor = _require_zstandard()
        with resolved_input.open("rb") as raw_handle:
            with compressor.ZstdDecompressor().stream_reader(raw_handle) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as handle:
                    payload = json.load(handle)
    return payload, CheckpointReadStats(
        input_path=resolved_input,
        file_format=resolved_format,
        compressed_bytes=compressed_bytes,
        json_load_s=perf_counter() - load_started_at,
    )


def _require_zstandard() -> Any:
    """Return the optional zstandard module or fail with a clear message."""
    if _zstandard is None:
        raise RuntimeError("zstandard is required for .json.zst checkpoint support")
    return _zstandard


@dataclass(frozen=True, slots=True)
class _EncodedCheckpointPayload:
    """Intermediate encoded checkpoint payload before optional compression."""

    json_bytes: bytes
    encoder: CheckpointJsonEncoder
    jsonable_s: float | None
    json_encode_s: float


def _dataclass_field_names(dataclass_type: type[object]) -> tuple[str, ...]:
    """Return cached dataclass field names for one payload type."""
    cached = _DATACLASS_FIELD_NAMES_BY_TYPE.get(dataclass_type)
    if cached is not None:
        return cached
    dataclass_fields = getattr(dataclass_type, "__dataclass_fields__", None)
    if dataclass_fields is None:
        return ()
    field_names = tuple(dataclass_fields)
    _DATACLASS_FIELD_NAMES_BY_TYPE[dataclass_type] = field_names
    return field_names


def _encode_checkpoint_payload(payload: object) -> _EncodedCheckpointPayload:
    """Return one whole encoded JSON payload without chunked writes."""
    if _orjson is not None:
        direct_encoded = _try_orjson_encode(payload, serialize_dataclass=True)
        if direct_encoded is not None:
            return _EncodedCheckpointPayload(
                json_bytes=direct_encoded[0],
                encoder="orjson",
                jsonable_s=None,
                json_encode_s=direct_encoded[1],
            )

    jsonable_started_at = perf_counter()
    jsonable_payload = checkpoint_payload_to_jsonable(payload)
    jsonable_s = perf_counter() - jsonable_started_at

    if _orjson is not None:
        json_bytes, json_encode_s = _orjson_encode(jsonable_payload)
        return _EncodedCheckpointPayload(
            json_bytes=json_bytes,
            encoder="orjson",
            jsonable_s=jsonable_s,
            json_encode_s=json_encode_s,
        )

    json_encode_started_at = perf_counter()
    json_bytes = json.dumps(jsonable_payload, indent=2, sort_keys=True).encode("utf-8")
    return _EncodedCheckpointPayload(
        json_bytes=json_bytes,
        encoder="stdlib",
        jsonable_s=jsonable_s,
        json_encode_s=perf_counter() - json_encode_started_at,
    )


def _orjson_encode(payload: object) -> tuple[bytes, float]:
    """Encode one payload with orjson and stable formatting options."""
    orjson_module = _require_orjson()
    started_at = perf_counter()
    json_bytes = orjson_module.dumps(
        payload,
        option=orjson_module.OPT_INDENT_2 | orjson_module.OPT_SORT_KEYS,
    )
    return json_bytes, perf_counter() - started_at


def _try_orjson_encode(
    payload: object,
    *,
    serialize_dataclass: bool,
) -> tuple[bytes, float] | None:
    """Try encoding with orjson and return ``None`` when the payload is unsupported."""
    orjson_module = _require_orjson()
    option = orjson_module.OPT_INDENT_2 | orjson_module.OPT_SORT_KEYS
    if serialize_dataclass:
        option |= orjson_module.OPT_SERIALIZE_DATACLASS
    started_at = perf_counter()
    try:
        json_bytes = orjson_module.dumps(payload, option=option)
    except TypeError:
        return None
    return json_bytes, perf_counter() - started_at


def _compress_checkpoint_bytes(
    json_bytes: bytes,
    file_format: CheckpointFileFormat,
) -> tuple[bytes, float | None]:
    """Compress encoded checkpoint bytes in one call when needed."""
    if file_format == "json":
        return json_bytes, None
    compress_started_at = perf_counter()
    if file_format == "json_gz":
        compressed_bytes = gzip.compress(json_bytes, compresslevel=6, mtime=0)
        return compressed_bytes, perf_counter() - compress_started_at
    compressor = _require_zstandard().ZstdCompressor()
    compressed_bytes = compressor.compress(json_bytes)
    return compressed_bytes, perf_counter() - compress_started_at


def _require_orjson() -> Any:
    """Return the optional orjson module or fail with a clear message."""
    if _orjson is None:
        raise RuntimeError("orjson is required for the fast checkpoint JSON encoder")
    return _orjson


__all__ = [
    "DEFAULT_CHECKPOINT_FILE_FORMAT",
    "CheckpointFileFormat",
    "CheckpointJsonEncoder",
    "CheckpointReadStats",
    "CheckpointWriteStats",
    "checkpoint_cli_name",
    "checkpoint_file_format_from_path",
    "checkpoint_file_suffix",
    "checkpoint_format_from_cli_name",
    "checkpoint_output_path",
    "checkpoint_path_for_generation",
    "checkpoint_payload_to_jsonable",
    "default_checkpoint_file_suffix",
    "load_checkpoint_json_payload",
    "parse_generation_checkpoint_name",
    "resolve_latest_generation_checkpoint_path",
    "write_checkpoint_json_payload",
]