"""Persisted input parity and bounded refusal, without a runtime intermediary."""

# ruff: noqa: D103
import json
from compression import zstd
from dataclasses import replace
from time import monotonic
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from anemone.checkpoints import (
    ShardedCheckpointReadLimits,
    build_search_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    load_sharded_search_checkpoint,
    read_sharded_checkpoint_manifest,
    write_sharded_search_checkpoint,
    write_sharded_search_checkpoint_from_search,
)
from anemone.checkpoints.io import checkpoint_payload_to_jsonable
from anemone.prepared_evaluator import (
    prepare_evaluator_context,
    stage_evaluator_context,
)
from tests.test_checkpoint_sharded import _small_checkpoint_payload
from tests.test_prepared_evaluator import (
    IDENTITY,
    INDEX_RECIPES,
    Candidate,
    Codec,
    Incumbent,
    limits,
    native_bytes,
)
from tests.test_prepared_evaluator_continuation import pinned, source_case


def read_limits(**changes: Any) -> ShardedCheckpointReadLimits:
    return replace(
        ShardedCheckpointReadLimits(
            deadline_monotonic=monotonic() + 30,
            max_nodes=512,
            max_branches=4096,
            max_shards=1024,
            max_compressed_bytes=8 * 2**20,
            max_uncompressed_bytes=8 * 2**20,
            max_record_bytes=2 * 2**20,
        ),
        **changes,
    )


def manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_bytes())


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    (path / "manifest.json").write_text(json.dumps(data))


def rewrite(path: Path, family: str, transform: Any) -> None:
    data = manifest(path)
    shard = next(s for s in data["shards"] if s["kind"] == family)
    target = path / shard["path"]
    rows = [json.loads(line) for line in target.read_bytes().splitlines()]
    transform(rows)
    target.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    # Exercise content validation independently of optional integrity declarations.
    shard["compressed_bytes"] = shard["uncompressed_bytes"] = shard["sha256"] = None
    shard["record_count"] = len(rows)
    save_manifest(path, data)


@pytest.mark.parametrize("layout", ["split", "node_records"])
@pytest.mark.parametrize("encoding", ["jsonl", "jsonl_zst"])
@pytest.mark.parametrize("index", INDEX_RECIPES)
def test_typed_roundtrip_all_fields(
    tmp_path: Path, layout: Any, encoding: Any, index: Any
) -> None:
    live, _, _ = source_case(rollout=True, steps=3, index_computation=index)
    original = build_search_checkpoint_payload(live, state_codec=Codec())
    write_sharded_search_checkpoint(
        original, tmp_path, layout=layout, encoding=encoding, node_count_per_shard=7
    )
    result = load_sharded_search_checkpoint(tmp_path, limits=read_limits())
    # Native JSON-compatible form accounts only for tuple-to-list wire encoding.
    assert checkpoint_payload_to_jsonable(result) == checkpoint_payload_to_jsonable(
        original
    )
    assert result == load_sharded_search_checkpoint(tmp_path)
    assert all(
        (n.exploration_index is None) == (index is None) for n in result.tree.nodes
    )
    assert result.rollout_rng_state is not None and result.rng_state is not None
    assert result.selector_state.selection_step_count == 3


def test_record_order_join_and_no_hidden_runtime(
    tmp_path: Path, monkeypatch: Any
) -> None:
    payload = _small_checkpoint_payload()
    write_sharded_search_checkpoint(payload, tmp_path, layout="split", encoding="jsonl")
    for family in ("state_payloads", "node_runtime", "node_shells"):
        rewrite(tmp_path, family, lambda rows: rows.reverse())

    def forbidden(*args: Any, **kwargs: Any) -> None:
        message = "materializer constructed a runtime or invoked domain code"
        raise AssertionError(message)

    from anemone.node_selector.linoo import Linoo
    from anemone.tree_exploration import TreeExploration

    for cls in (Codec, Candidate, Linoo, TreeExploration):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(Codec, "load_anchor_ref", forbidden)
    monkeypatch.setattr(Codec, "load_child_from_delta", forbidden)
    monkeypatch.setattr(Candidate, "evaluate_batch_items", forbidden)
    result = load_sharded_search_checkpoint(tmp_path, limits=read_limits())
    assert result == replace(
        payload, tree=replace(payload.tree, nodes=list(reversed(payload.tree.nodes)))
    )


@pytest.mark.parametrize("family", ["node_shells", "state_payloads", "node_runtime"])
@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "extra", "wrong-id", "missing-field"]
)
def test_family_coverage_refusals(tmp_path: Path, family: str, fault: str) -> None:
    write_sharded_search_checkpoint(
        _small_checkpoint_payload(), tmp_path, layout="split", encoding="jsonl"
    )

    def corrupt(rows: Any) -> None:
        if fault == "duplicate":
            rows[1] = rows[0]
        elif fault == "missing":
            rows.pop()
        elif fault == "extra":
            rows.append(dict(rows[0], node_id=100))
        elif fault == "wrong-id":
            rows[1]["node_id"] = 100
        else:
            rows[0].pop(next(key for key in rows[0] if key != "node_id"))

    rewrite(tmp_path, family, corrupt)
    with pytest.raises(ValueError):
        load_sharded_search_checkpoint(tmp_path, limits=read_limits())


@pytest.mark.parametrize(
    "fault",
    [
        "mixed",
        "incomplete",
        "summary",
        "singleton",
        "record-count",
        "root",
        "branch-total",
        "duplicate-path",
        "escape",
        "symlink",
    ],
)
def test_manifest_and_conflict_refusals(tmp_path: Path, fault: str) -> None:
    path = tmp_path / "input"
    write_sharded_search_checkpoint(
        _small_checkpoint_payload(), path, layout="split", encoding="jsonl"
    )
    data = manifest(path)
    if fault == "summary":
        rewrite(
            path,
            "node_shells",
            lambda rows: rows[0].update(state_summary={"wrong": True}),
        )
    elif fault == "root":
        shard = next(s for s in data["shards"] if s["kind"] == "metadata")
        target = path / shard["path"]
        obj = json.loads(zstd.decompress(target.read_bytes()))
        obj["root_node_id"] = 999
        target.write_bytes(zstd.compress(json.dumps(obj).encode()))
        shard["compressed_bytes"] = shard["uncompressed_bytes"] = None
    elif fault == "mixed":
        data["shards"].append(
            dict(data["shards"][1], kind="node_records", path="extra.jsonl")
        )
    elif fault == "incomplete":
        data["shards"] = [s for s in data["shards"] if s["kind"] != "node_runtime"]
    elif fault == "singleton":
        data["shards"][0]["record_count"] = 2
    elif fault == "record-count":
        data["shards"][1]["record_count"] += 1
    elif fault == "branch-total":
        data["total_branch_count"] += 1
    elif fault == "duplicate-path":
        data["shards"][1]["path"] = data["shards"][0]["path"]
    elif fault == "escape":
        data["shards"][1]["path"] = "../outside"
    elif fault == "symlink":
        target = path / data["shards"][1]["path"]
        outside = tmp_path / "outside"
        target.rename(outside)
        target.symlink_to(outside)
    if fault != "summary":
        save_manifest(path, data)
    with pytest.raises(ValueError):
        load_sharded_search_checkpoint(path, limits=read_limits())


@pytest.mark.parametrize(
    "name",
    [
        "max_nodes",
        "max_branches",
        "max_shards",
        "max_compressed_bytes",
        "max_uncompressed_bytes",
        "max_record_bytes",
    ],
)
def test_every_limit(tmp_path: Path, name: str) -> None:
    write_sharded_search_checkpoint(
        _small_checkpoint_payload(), tmp_path, layout="split"
    )
    with pytest.raises(ValueError, match=name):
        load_sharded_search_checkpoint(tmp_path, limits=read_limits(**{name: 1}))


@pytest.mark.parametrize(
    "fault",
    ["record", "decompressed", "truncated", "trailing", "duplicate-key", "digest"],
)
def test_streaming_byte_guards(tmp_path: Path, fault: str) -> None:
    write_sharded_search_checkpoint(
        _small_checkpoint_payload(), tmp_path, layout="split"
    )
    data = manifest(tmp_path)
    shard = next(s for s in data["shards"] if s["kind"] == "state_payloads")
    target = tmp_path / shard["path"]
    raw = zstd.decompress(target.read_bytes())
    budget = read_limits()
    if fault in ("record", "decompressed"):
        raw = b'{"padding":"' + b"x" * (1024 * 1024) + b'"}\n'
        budget = read_limits(
            **(
                {"max_record_bytes": 8192}
                if fault == "record"
                else {"max_uncompressed_bytes": 8192}
            )
        )
        target.write_bytes(zstd.compress(raw))
    elif fault == "truncated":
        target.write_bytes(target.read_bytes()[:-1])
    elif fault == "trailing":
        target.write_bytes(target.read_bytes() + b"junk")
    elif fault == "duplicate-key":
        target.write_bytes(zstd.compress(b'{"node_id":0,"node_id":1}\n'))
    else:
        shard["sha256"] = "0" * 64
    shard["compressed_bytes"] = shard["uncompressed_bytes"] = None
    save_manifest(tmp_path, data)
    with pytest.raises((ValueError, zstd.ZstdError)):
        load_sharded_search_checkpoint(tmp_path, limits=budget)


@pytest.mark.parametrize("phase", ["initial", "during-read", "deadline"])
def test_one_deadline_and_cancellation(tmp_path: Path, phase: str) -> None:
    write_sharded_search_checkpoint(
        _small_checkpoint_payload(), tmp_path, layout="split"
    )
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return phase == "initial" or calls > 25

    budget = (
        read_limits(deadline_monotonic=monotonic() - 1)
        if phase == "deadline"
        else read_limits(cancelled=cancelled)
    )
    with pytest.raises(ValueError, match=r"deadline|cancelled"):
        load_sharded_search_checkpoint(tmp_path, limits=budget)


@pytest.mark.parametrize("index", INDEX_RECIPES)
def test_original_split_input_prepared_install_and_continuation(
    tmp_path: Path, index: Any
) -> None:
    from anemone.node_evaluation.tree.single_agent.factory import (
        NodeMaxEvaluationFactory,
    )
    from tests.morpion_refresh_helpers import Dynamics, State

    live, config, recipe = source_case(rollout=True, steps=6, index_computation=index)
    original = build_search_checkpoint_payload(live, state_codec=Codec())
    nodes = {n.id: n for n in live._iter_nodes_by_depth()}
    record = next(r for r in original.tree.nodes if nodes[r.node_id].parent_count() > 1)
    alternate = next(
        p
        for p, _ in nodes[record.node_id].iter_parent_items()
        if p.id != record.state_payload.state_parent_node_id
    )
    record.parent_node_id = alternate.id
    parent = next(r for r in original.tree.nodes if r.node_id == alternate.id)
    record.branch_from_parent = next(
        link.branch_key
        for link in parent.linked_children
        if link.child_node_id == record.node_id
    )
    record.parent_node_order = [alternate.id] + [
        p.id
        for p, _ in nodes[record.node_id].iter_parent_items()
        if p.id != alternate.id
    ]
    checkpoint = tmp_path / "source"
    write_sharded_search_checkpoint(
        original, checkpoint, layout="split", node_count_per_shard=11
    )
    inventory = {
        str(p.relative_to(checkpoint)): p.read_bytes()
        for p in checkpoint.rglob("*")
        if p.is_file()
    }
    materialized = load_sharded_search_checkpoint(checkpoint, limits=read_limits())
    assert checkpoint_payload_to_jsonable(
        materialized
    ) == checkpoint_payload_to_jsonable(original)
    source = pinned(materialized, recipe)
    candidate = Candidate({i: float(i + 10) for i in range(512)})
    context = prepare_evaluator_context(
        source, evaluator=candidate, evaluation_identity=IDENTITY, limits=limits()
    )

    def cold() -> Any:
        return load_search_from_sharded_checkpoint(
            checkpoint,
            state_codec=Codec(),
            dynamics=Dynamics(),
            args=config,
            state_type=State,
            master_state_value_evaluator=Incumbent(),
            node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        )

    owner, reference = cold(), cold()
    reference.refresh_with_evaluator(candidate, scope="all")
    stage_evaluator_context(
        owner,
        current_source=source,
        prepared=context,
        evaluator=candidate,
        evaluation_identity=IDENTITY,
        limits=limits(),
    ).install()
    assert native_bytes(owner) == native_bytes(reference)
    for name, search in (("installed", owner), ("reference", reference)):
        write_sharded_search_checkpoint_from_search(
            search,
            tmp_path / name,
            state_codec=Codec(),
            layout="split",
            node_count_per_shard=7,
        )
    expected = {
        p.relative_to(tmp_path / "reference"): p.read_bytes()
        for p in (tmp_path / "reference").rglob("*")
        if p.is_file()
    }
    assert expected == {
        p.relative_to(tmp_path / "installed"): p.read_bytes()
        for p in (tmp_path / "installed").rglob("*")
        if p.is_file()
    }
    for _ in range(2):
        count = owner.tree.branch_count
        owner.step()
        reference.step()
        assert owner.tree.branch_count > count
        assert native_bytes(owner) == native_bytes(reference)
    assert inventory == {
        str(p.relative_to(checkpoint)): p.read_bytes()
        for p in checkpoint.rglob("*")
        if p.is_file()
    }


def test_invalid_history_survives_materialization_and_freeze_refuses(
    tmp_path: Path,
) -> None:
    live, _, recipe = source_case(steps=2, index_computation=None)
    payload = build_search_checkpoint_payload(live, state_codec=Codec())
    payload.selector_state.node_states.clear()
    write_sharded_search_checkpoint(payload, tmp_path, layout="split")
    result = load_sharded_search_checkpoint(tmp_path, limits=read_limits())
    assert result.selector_state == payload.selector_state
    with pytest.raises(ValueError, match="Linoo"):
        pinned(result, recipe)


@pytest.mark.parametrize("index", INDEX_RECIPES)
def test_terminal_and_direct_certificate_fields_survive(
    tmp_path: Path, index: Any
) -> None:
    from tests.test_prepared_evaluator_continuation import endgame

    live, _, _ = source_case(steps=1, state=endgame(), index_computation=index)
    root = live.tree.root_node.tree_evaluation
    root.direct_value = root.tree_value
    original = build_search_checkpoint_payload(live, state_codec=Codec())
    assert original.tree.nodes[0].evaluation.direct_value.certainty == "FORCED"
    write_sharded_search_checkpoint(original, tmp_path, layout="split")
    result = load_sharded_search_checkpoint(tmp_path, limits=read_limits())
    assert checkpoint_payload_to_jsonable(result) == checkpoint_payload_to_jsonable(
        original
    )


def test_actual_bytes_inclusive_limits_and_multiframe(tmp_path: Path) -> None:
    from anemone.checkpoints.sharded_read import _ShardReader

    original = _small_checkpoint_payload()
    write_sharded_search_checkpoint(original, tmp_path, layout="split")
    data = manifest(tmp_path)
    shard = next(s for s in data["shards"] if s["kind"] == "state_payloads")
    target = tmp_path / shard["path"]
    raw = zstd.decompress(target.read_bytes())
    cut = len(raw) // 2
    target.write_bytes(zstd.compress(raw[:cut]) + zstd.compress(raw[cut:]))
    shard["compressed_bytes"] = len(target.read_bytes())
    save_manifest(tmp_path, data)
    reader = _ShardReader(tmp_path, read_limits())
    reader.mapping("manifest.json")
    for ref in read_sharded_checkpoint_manifest(tmp_path / "manifest.json").shards:
        if ref.encoding.startswith("jsonl"):
            list(reader.records(ref))
        else:
            reader.mapping(ref.path, ref)
    budget = read_limits(
        max_compressed_bytes=reader.compressed_bytes,
        max_uncompressed_bytes=reader.uncompressed_bytes,
        max_record_bytes=reader.peak_record_bytes,
    )
    assert load_sharded_search_checkpoint(tmp_path, limits=budget) == original
    for field in ("max_compressed_bytes", "max_uncompressed_bytes", "max_record_bytes"):
        with pytest.raises(ValueError, match=field):
            load_sharded_search_checkpoint(
                tmp_path, limits=replace(budget, **{field: getattr(budget, field) - 1})
            )


def test_deadline_expires_during_read(tmp_path: Path, monkeypatch: Any) -> None:
    from anemone.checkpoints import sharded_read_limits

    write_sharded_search_checkpoint(
        _small_checkpoint_payload(), tmp_path, layout="split"
    )
    ticks = 0

    def clock() -> float:
        nonlocal ticks
        ticks += 1
        return ticks / 100

    monkeypatch.setattr(sharded_read_limits, "monotonic", clock)
    with pytest.raises(ValueError, match="deadline"):
        load_sharded_search_checkpoint(
            tmp_path, limits=read_limits(deadline_monotonic=0.25)
        )
    assert ticks >= 25


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_nodes", 0),
        ("max_shards", True),
        ("max_record_bytes", -1),
        ("deadline_monotonic", float("inf")),
    ],
)
def test_invalid_budget_refuses(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="invalid_read_limits"):
        read_limits(**{field: value})


@pytest.mark.parametrize(
    "fault",
    [
        "null-list",
        "missing-evaluation-field",
        "unknown-evaluation-field",
        "metadata-count",
    ],
)
def test_split_never_synthesizes_missing_fields(tmp_path: Path, fault: str) -> None:
    write_sharded_search_checkpoint(
        _small_checkpoint_payload(), tmp_path, layout="split", encoding="jsonl"
    )
    if fault == "metadata-count":
        data = manifest(tmp_path)
        shard = next(s for s in data["shards"] if s["kind"] == "metadata")
        target = tmp_path / shard["path"]
        raw = json.loads(zstd.decompress(target.read_bytes()))
        raw["node_count"] += 1
        target.write_bytes(zstd.compress(json.dumps(raw).encode()))
        shard["compressed_bytes"] = shard["uncompressed_bytes"] = None
        save_manifest(tmp_path, data)
    else:

        def corrupt(rows: Any) -> None:
            if fault == "null-list":
                rows[0]["unopened_branches"] = None
            else:
                from anemone.checkpoints import NodeEvaluationCheckpointPayload

                evaluation = checkpoint_payload_to_jsonable(
                    NodeEvaluationCheckpointPayload()
                )
                if fault == "missing-evaluation-field":
                    evaluation.pop("direct_value")
                else:
                    evaluation["unknown"] = 1
                rows[0]["evaluation"] = evaluation

        rewrite(tmp_path, "node_runtime", corrupt)
    with pytest.raises(ValueError):
        load_sharded_search_checkpoint(tmp_path, limits=read_limits())
