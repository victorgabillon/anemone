"""Errors raised while reading or writing sharded checkpoints."""

from __future__ import annotations


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
    def non_positive_restore_batch_size(cls) -> ShardedCheckpointManifestError:
        """Return the error for an invalid restore batch size."""
        return cls("sharded checkpoint restore batch_size must be positive.")

    @classmethod
    def unsupported_jsonl_encoding(cls) -> ShardedCheckpointManifestError:
        """Return the error for an unsupported JSONL shard encoding."""
        return cls(
            "sharded checkpoint node records require jsonl or jsonl_zst encoding."
        )

    @classmethod
    def unsupported_shard_kind(cls) -> ShardedCheckpointManifestError:
        """Return the error for an unsupported shard kind."""
        return cls("sharded checkpoint manifest contains an unsupported shard kind.")

    @classmethod
    def unsupported_shard_encoding(cls) -> ShardedCheckpointManifestError:
        """Return the error for an unsupported shard encoding."""
        return cls(
            "sharded checkpoint manifest contains an unsupported shard encoding."
        )

    @classmethod
    def unsupported_layout(cls) -> ShardedCheckpointManifestError:
        """Return the error for an unsupported sharded writer layout."""
        return cls("sharded checkpoint layout must be 'node_records' or 'split'.")

    @classmethod
    def missing_metadata_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for a missing metadata shard."""
        return cls(
            "sharded checkpoint manifest must contain exactly one metadata shard."
        )

    @classmethod
    def duplicate_metadata_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for multiple metadata shards."""
        return cls("sharded checkpoint manifest contains multiple metadata shards.")

    @classmethod
    def missing_node_records_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for missing node-record shards."""
        return cls("sharded checkpoint manifest must contain node-record shards.")

    @classmethod
    def missing_node_shells_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for missing split-layout node-shell shards."""
        return cls("split sharded checkpoint manifest must contain node-shell shards.")

    @classmethod
    def missing_state_payloads_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for missing split-layout state-payload shards."""
        return cls(
            "split sharded checkpoint manifest must contain state-payload shards."
        )

    @classmethod
    def missing_node_runtime_shard(cls) -> ShardedCheckpointManifestError:
        """Return the error for missing split-layout node-runtime shards."""
        return cls(
            "split sharded checkpoint manifest must contain node-runtime shards."
        )

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
        return cls(
            "sharded checkpoint shard path resolves outside the checkpoint root."
        )

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
        return cls("sharded checkpoint shard paths must be normalized relative paths.")
