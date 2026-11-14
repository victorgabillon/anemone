"""Explicit cooperative limits for complete typed checkpoint materialization."""

# Exact integer bounds reject bool and subclasses; do not broaden admission.

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import monotonic

from .sharded_errors import ShardedCheckpointManifestError


@dataclass(frozen=True, slots=True)
class ShardedCheckpointReadLimits:
    """Bound all input families under one deadline; callers still contain RSS/time.

    Compressed bytes count actual file bytes, including plain JSON and manifest.
    Uncompressed bytes count all decoded bytes, including whitespace. The record
    bound also applies to each singleton JSON document and the manifest itself.
    """

    deadline_monotonic: float
    max_nodes: int
    max_branches: int
    max_shards: int
    max_compressed_bytes: int
    max_uncompressed_bytes: int
    max_record_bytes: int
    cancelled: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        """Reject invalid or unbounded declarations at construction."""
        values = (
            self.max_nodes,
            self.max_branches,
            self.max_shards,
            self.max_compressed_bytes,
            self.max_uncompressed_bytes,
            self.max_record_bytes,
        )
        if not isfinite(self.deadline_monotonic) or any(
            type(value) is not int or value <= 0  # pylint: disable=unidiomatic-typecheck
            for value in values
        ):
            raise ShardedCheckpointManifestError.invalid_input("invalid_read_limits")

    def check(self) -> None:
        """Check cancellation and the original absolute deadline."""
        if self.cancelled is not None and self.cancelled():
            raise ShardedCheckpointManifestError.invalid_input("read_cancelled")
        if monotonic() >= self.deadline_monotonic:
            raise ShardedCheckpointManifestError.invalid_input(
                "read_deadline_exhausted"
            )
