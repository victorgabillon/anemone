"""Streaming construction preserves the complete format-3 checkpoint contract."""
# ruff: noqa: D103

import hashlib
import subprocess
import sys
import time
from functools import partial
from heapq import heappush
from pathlib import Path
from random import Random
from typing import Any

import pytest

from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
    load_search_from_sharded_checkpoint,
    load_sharded_search_checkpoint,
    write_sharded_search_checkpoint,
    write_sharded_search_checkpoint_from_search,
)
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from tests import test_morpion_ordered_checkpoint_continuation as morpion
from tests.fake_yaml_game import FakeYamlDynamics, MasterStateValueEvaluatorFromYaml
from tests.scientific_refresh_helpers import reference, snapshot
from tests.test_checkpoint_load import _ConcreteFakeYamlState
from tests.test_checkpoint_ordered_continuation import _VALUES, fixture, ordered
from tests.test_refresh_scopes import _build_args


def files(path: Path) -> dict[str, str]:
    """Compare every manifest, metadata and shard byte, with no field exclusions."""
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }


def restore_small(path: Path, codec: Any) -> Any:
    return load_search_from_sharded_checkpoint(
        path,
        state_codec=codec,
        dynamics=FakeYamlDynamics(),
        args=_build_args(),
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(_VALUES),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )


@pytest.mark.parametrize("layout", ["node_records", "split"])
@pytest.mark.parametrize("encoding", ["jsonl", "jsonl_zst"])
@pytest.mark.parametrize("encoder", ["stdlib", "orjson"])
@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_stream_matches_every_typed_checkpoint_byte_and_repeated_restore(
    layout: Any, encoding: Any, encoder: Any, batch_size: int, tmp_path: Path
) -> None:
    search, codec = fixture()
    for index in range(3):
        payload = build_search_checkpoint_payload(search, state_codec=codec)
        expected = checkpoint_payload_to_jsonable(payload)
        kwargs = dict(
            layout=layout,
            encoding=encoding,
            encoder=encoder,
            node_count_per_shard=batch_size,
            generation=37,
        )
        baseline, streamed = tmp_path / f"base-{index}", tmp_path / f"stream-{index}"
        a = write_sharded_search_checkpoint(payload, baseline, **kwargs)
        b = write_sharded_search_checkpoint_from_search(
            search, streamed, state_codec=codec, **kwargs
        )
        assert a == b
        assert files(baseline) == files(streamed)
        if layout == "node_records":
            assert (
                checkpoint_payload_to_jsonable(load_sharded_search_checkpoint(streamed))
                == expected
            )
        restored = restore_small(streamed, codec)
        assert ordered(restored) == ordered(search)
        assert snapshot(restored) == snapshot(search) == reference(restored)
        assert (
            checkpoint_payload_to_jsonable(
                build_search_checkpoint_payload(restored, state_codec=codec)
            )
            == expected
        )
        search = restored


@pytest.mark.parametrize("layout", ["node_records", "split"])
def test_stream_writes_before_finishing_nodes_and_resets_batch_caches(
    layout: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anemone.checkpoints import build, build_node_payloads, sharded_encode

    search, codec = fixture()
    built = 0
    counts_at_write = []
    original_build = build_node_payloads._build_node_payload
    original_write = sharded_encode._write_jsonl_record_shard

    def build_node(**kwargs: Any) -> Any:
        nonlocal built
        assert not kwargs["context"].atom_serialization_cache
        assert not kwargs["context"].value_identity_serialization_cache
        built += 1
        return original_build(**kwargs)

    def write_shard(*args: Any, **kwargs: Any) -> Any:
        counts_at_write.append(built)
        return original_write(*args, **kwargs)

    def refuse_materialization(**_kwargs: Any) -> Any:
        pytest.fail("Streaming must not construct a full typed tree")

    monkeypatch.setattr(build, "_build_tree_payload", refuse_materialization)
    monkeypatch.setattr(build_node_payloads, "_build_node_payload", build_node)
    monkeypatch.setattr(sharded_encode, "_write_jsonl_record_shard", write_shard)
    if layout == "node_records":
        from anemone.checkpoints import sharded_io

        monkeypatch.setattr(sharded_io, "_write_jsonl_record_shard", write_shard)
    write_sharded_search_checkpoint_from_search(
        search, tmp_path, state_codec=codec, layout=layout, node_count_per_shard=1
    )
    assert counts_at_write[0] == 1 < built


@pytest.mark.parametrize("layout", ["node_records", "split"])
def test_interrupted_stream_has_no_complete_manifest(
    layout: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anemone.checkpoints import build_node_payloads

    search, codec = fixture()
    original = build_node_payloads._build_node_payload
    count = 0

    def interrupt(**kwargs: Any) -> Any:
        nonlocal count
        count += 1
        if count == 3:
            message = "injected serialization interruption"
            raise OSError(message)
        return original(**kwargs)

    monkeypatch.setattr(build_node_payloads, "_build_node_payload", interrupt)
    with pytest.raises(OSError, match="injected"):
        write_sharded_search_checkpoint_from_search(
            search, tmp_path, state_codec=codec, layout=layout, node_count_per_shard=1
        )
    assert any(tmp_path.rglob("*.zst"))
    assert not (tmp_path / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        restore_small(tmp_path, codec)


@pytest.mark.parametrize("typed", [False, True])
def test_output_cannot_overwrite_a_complete_or_partial_checkpoint(
    typed: bool, tmp_path: Path
) -> None:
    search, codec = fixture()
    write_sharded_search_checkpoint_from_search(search, tmp_path, state_codec=codec)
    before = files(tmp_path)
    with pytest.raises(FileExistsError, match="must be empty"):
        if typed:
            write_sharded_search_checkpoint(
                build_search_checkpoint_payload(search, state_codec=codec), tmp_path
            )
        else:
            write_sharded_search_checkpoint_from_search(
                search, tmp_path, state_codec=codec
            )
    assert files(tmp_path) == before
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(FileExistsError, match="must be empty"):
        write_sharded_search_checkpoint_from_search(search, tmp_path, state_codec=codec)


def restore_morpion_stream(
    search: Any,
    args: Any,
    codec: Any,
    storage: str,
    path: Path,
    evaluator: Any,
    *,
    layout: Any,
) -> Any:
    assert storage == "sharded"
    write_sharded_search_checkpoint_from_search(
        search, path, state_codec=codec, layout=layout, node_count_per_shard=97
    )
    return load_search_from_sharded_checkpoint(
        path,
        state_codec=codec,
        dynamics=morpion.Dynamics(),
        args=args,
        state_type=morpion.State,
        master_state_value_evaluator=evaluator,
        random_generator=Random(999),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )


@pytest.mark.parametrize("layout", ["node_records", "split"])
@pytest.mark.parametrize("rollout", [False, True])
def test_original_real_morpion_witness_unchanged_with_streaming_save(
    layout: Any, rollout: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_runtime = morpion.make_runtime

    def runtime_with_stale_entry(enabled: bool) -> Any:
        search, args = original_runtime(enabled)
        heap = search.node_selector.base._candidate_heap
        for entries in heap.candidates_by_depth.values():
            for priority, node_id, version in tuple(entries):
                if version == heap.candidate_versions_by_node_id.get(node_id):
                    heappush(entries, (priority - 1.0, node_id, version - 1))
                    return search, args
        pytest.fail("Witness needs both current and stale heap entries")

    monkeypatch.setattr(morpion, "make_runtime", runtime_with_stale_entry)
    monkeypatch.setattr(
        morpion, "restore", partial(restore_morpion_stream, layout=layout)
    )
    morpion.test_actual_1k_repeated_checkpoints_through_30_growth_steps(
        "sharded", rollout, tmp_path, monkeypatch
    )


@pytest.mark.parametrize("layout", ["node_records", "split"])
def test_terminal_and_forced_values_preserve_the_complete_payload(
    layout: Any, tmp_path: Path
) -> None:
    from valanga.evaluations import Certainty

    from tests.scientific_refresh_helpers import nodes

    search, codec = fixture()
    search.step()
    assert any(
        n.tree_evaluation.direct_value.certainty == Certainty.TERMINAL
        for n in nodes(search)
    )
    assert (
        search.tree.root_node.tree_evaluation.effective_value.certainty
        == Certainty.FORCED
    )
    expected = checkpoint_payload_to_jsonable(
        build_search_checkpoint_payload(search, state_codec=codec)
    )
    write_sharded_search_checkpoint_from_search(
        search, tmp_path, state_codec=codec, node_count_per_shard=1, layout=layout
    )
    restored = restore_small(tmp_path, codec)
    assert snapshot(restored) == snapshot(search) == reference(restored)
    assert (
        checkpoint_payload_to_jsonable(
            build_search_checkpoint_payload(restored, state_codec=codec)
        )
        == expected
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX process signals")
def test_process_death_mid_shard_sequence_is_not_a_complete_checkpoint(
    tmp_path: Path,
) -> None:
    import signal

    script = """
import os, signal, sys
from pathlib import Path
from anemone.checkpoints import build_node_payloads, write_sharded_search_checkpoint_from_search
from tests.test_checkpoint_ordered_continuation import fixture
root = Path(sys.argv[1])
original = build_node_payloads._build_node_payload
count = 0
def stop(**kwargs):
    global count
    count += 1
    if count == 3:
        (root / "ready").write_text("two nodes written")
        os.kill(os.getpid(), signal.SIGSTOP)
    return original(**kwargs)
build_node_payloads._build_node_payload = stop
search, codec = fixture()
write_sharded_search_checkpoint_from_search(search, root / "partial", state_codec=codec, layout="split", node_count_per_shard=1)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)])
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready").exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.kill()
        assert process.wait(timeout=10) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    assert any((tmp_path / "partial").rglob("*.zst"))
    assert not (tmp_path / "partial" / "manifest.json").exists()
    _, codec = fixture()
    with pytest.raises(FileNotFoundError):
        restore_small(tmp_path / "partial", codec)
