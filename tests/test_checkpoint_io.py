"""Tests for checkpoint JSON I/O helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anemone.checkpoints import (
    checkpoint_output_path,
    load_checkpoint_json_payload,
    write_checkpoint_json_payload,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def test_load_checkpoint_json_payload_logs_read_phases(tmp_path: Path) -> None:
    """Checkpoint reads should expose stream-open and raw-decode boundaries."""
    checkpoint_path = checkpoint_output_path(tmp_path / "checkpoint", file_format="json")
    write_checkpoint_json_payload({"tree": {"nodes": [1, 2]}}, checkpoint_path)
    phases: list[tuple[str, dict[str, object]]] = []

    def log_phase(phase: str, metadata: Mapping[str, object]) -> None:
        phases.append((phase, dict(metadata)))

    payload, stats = load_checkpoint_json_payload(
        checkpoint_path,
        read_phase_logger=log_phase,
    )

    assert payload == {"tree": {"nodes": [1, 2]}}
    assert stats.file_format == "json"
    assert [phase for phase, _metadata in phases] == [
        "after_checkpoint_file_read_or_stream_open",
        "after_raw_json_decode",
    ]
    stream_metadata = phases[0][1]
    raw_metadata = phases[1][1]
    assert "raw_payload" not in stream_metadata
    assert raw_metadata["raw_payload"] is payload
    assert raw_metadata["compressed_bytes"] == checkpoint_path.stat().st_size
