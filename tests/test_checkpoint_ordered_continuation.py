"""First-round-trip ordered DAG and representative/codec-parent separation."""
# ruff: noqa: D103

from pathlib import Path
from typing import Any

import pytest

from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
    load_checkpoint_json_payload,
    load_search_from_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    write_checkpoint_json_payload,
    write_sharded_search_checkpoint,
)
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from tests.fake_yaml_game import FakeYamlDynamics, MasterStateValueEvaluatorFromYaml
from tests.scientific_refresh_helpers import nodes, reference, runtime, snapshot
from tests.test_checkpoint_load import (
    _ConcreteFakeYamlState,
    _SelectiveDeltaCheckpointCodec,
)
from tests.test_checkpoint_rollout_rng import _decode_json
from tests.test_refresh_scopes import _build_args

_GRAPH = {0: [1, 2], 1: [3, 3], 2: [3], 3: [4], 4: []}
_VALUES = {i: float(i) for i in _GRAPH}


def ordered(search: Any) -> list[Any]:
    """Preserve actual map order; branch labels on each incoming edge are sets."""
    return [
        (
            n.id,
            [(b, c.id if c is not None else None) for b, c in n.iter_child_links()],
            [(p.id, sorted(bs, key=repr)) for p, bs in n.iter_parent_items()],
        )
        for n in nodes(search)
    ]


def fixture() -> tuple[Any, Any]:
    """Choose opposite graph/codec parents and a non-sorted live child order."""
    search = runtime(_GRAPH, _VALUES, steps=2)
    by_tag = {n.tag: n for n in nodes(search)}
    child = by_tag[3]
    parents = child.parent_nodes
    entries = sorted(parents.items(), key=lambda item: item[0].tag)
    parents.clear()
    parents.update(entries)
    assert [p.tag for p in child.parent_nodes] == [1, 2]
    for n in nodes(search):
        links = list(n.iter_child_links())
        if len(links) > 1:
            n.clear_child_links()
            for b, c in sorted(links, reverse=True):
                n.set_child_for_branch(b, c)
    codec = _SelectiveDeltaCheckpointCodec(_GRAPH, {(1, 3)})
    return search, codec


def roundtrip(search: Any, codec: Any, storage: str, directory: Path) -> Any:
    """Use the existing typed, JSON and streaming checkpoint loaders."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = build_search_checkpoint_payload(search, state_codec=codec)
    kwargs = dict(
        state_codec=codec,
        dynamics=FakeYamlDynamics(),
        args=_build_args(),
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(_VALUES),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    if storage == "typed":
        return load_search_from_checkpoint_payload(payload, **kwargs)
    if storage == "json-zst":
        path = directory / "checkpoint.json.zst"
        write_checkpoint_json_payload(payload, path)
        return load_search_from_checkpoint_payload(
            _decode_json(load_checkpoint_json_payload(path)[0]), **kwargs
        )
    write_sharded_search_checkpoint(
        payload,
        directory,
        node_count_per_shard=1,
        layout="split" if storage == "split" else "node_records",
    )
    return load_search_from_sharded_checkpoint(directory, **kwargs)


@pytest.mark.parametrize("storage", ["typed", "json-zst", "sharded", "split"])
def test_first_and_repeated_roundtrip_preserves_every_payload_field_and_order(
    storage: str, tmp_path: Path
) -> None:
    search, codec = fixture()
    initial = checkpoint_payload_to_jsonable(
        build_search_checkpoint_payload(search, state_codec=codec)
    )
    initial_order = ordered(search)
    assert snapshot(search) == reference(search)
    for index in range(3):
        search = roundtrip(search, codec, storage, tmp_path / str(index))
        assert ordered(search) == initial_order
        assert (
            checkpoint_payload_to_jsonable(
                build_search_checkpoint_payload(search, state_codec=codec)
            )
            == initial
        )
        assert snapshot(search) == reference(search)


@pytest.mark.parametrize("order", [None, [], [999], [999, 1000], [1, 1]])
@pytest.mark.parametrize("storage", ["typed", "sharded", "split"])
def test_invalid_multi_parent_order_refuses(
    order: Any, storage: str, tmp_path: Path
) -> None:
    search, codec = fixture()
    payload = build_search_checkpoint_payload(search, state_codec=codec)
    shared = next(
        n
        for n in payload.tree.nodes
        if n.node_id == next(n.id for n in nodes(search) if n.tag == 3)
    )
    shared.parent_node_order = order
    kwargs = dict(
        state_codec=codec,
        dynamics=FakeYamlDynamics(),
        args=_build_args(),
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(_VALUES),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    with pytest.raises(ValueError, match="parent order"):
        if storage == "typed":
            load_search_from_checkpoint_payload(payload, **kwargs)
        else:
            write_sharded_search_checkpoint(
                payload,
                tmp_path,
                node_count_per_shard=1,
                layout="split" if storage == "split" else "node_records",
            )
            load_search_from_sharded_checkpoint(tmp_path, **kwargs)


def test_version_two_payload_refuses_without_decoding() -> None:
    from dataclasses import replace

    search, codec = fixture()
    payload = replace(
        build_search_checkpoint_payload(search, state_codec=codec), format_version=2
    )
    codec.reset_counters()
    with pytest.raises(ValueError, match="Unsupported checkpoint format version 2"):
        load_search_from_checkpoint_payload(
            payload,
            state_codec=codec,
            dynamics=FakeYamlDynamics(),
            args=_build_args(),
            state_type=_ConcreteFakeYamlState,
            master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(_VALUES),
            node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        )
    assert not codec.loaded_anchor_node_ids and not codec.applied_delta_items


def test_null_child_slot_refuses_instead_of_losing_structural_information() -> None:
    search, codec = fixture()
    search.tree.root_node.set_child_for_branch(99, None)
    with pytest.raises(ValueError, match="null child-link"):
        build_search_checkpoint_payload(search, state_codec=codec)


@pytest.mark.parametrize("storage", ["typed", "json-zst", "sharded", "split"])
def test_equal_score_recommendation_observes_restored_child_order(
    storage: str, tmp_path: Path
) -> None:
    from random import Random

    from anemone.recommender_rule.recommender_rule import SoftmaxRule

    search, codec = fixture()
    recommender = SoftmaxRule(type="softmax", temperature=1.0)
    expected = recommender.sample(recommender.policy(search.tree.root_node), Random(0))
    restored = roundtrip(search, codec, storage, tmp_path)
    assert (
        recommender.sample(recommender.policy(restored.tree.root_node), Random(0))
        == expected
    )


def test_graph_representative_and_codec_parent_are_distinct_and_stable() -> None:
    from anemone.checkpoints import DeltaCheckpointStatePayload

    search, codec = fixture()
    payload = build_search_checkpoint_payload(search, state_codec=codec)
    child = next(n for n in nodes(search) if n.tag == 3)
    exported = next(n for n in payload.tree.nodes if n.node_id == child.id)
    assert exported.parent_node_order == [p.id for p in child.parent_nodes]
    assert exported.parent_node_id == next(iter(child.parent_nodes)).id
    assert isinstance(exported.state_payload, DeltaCheckpointStatePayload)
    assert exported.state_payload.state_parent_node_id != exported.parent_node_id


def test_checkpoint_heap_labels_are_canonical_without_mutating_runtime_history() -> (
    None
):
    from anemone.node_selector.linoo.checkpoint_state import (
        candidate_payloads_from_cache,
    )
    from anemone.node_selector.linoo.runtime_state import LinooNodeState

    states = {10: LinooNodeState(node_id=10, depth=2, status="frontier")}
    heap = {2: [(-9.0, 10, 7), (-100.0, 10, 6)]}
    versions = {10: 7}
    first = candidate_payloads_from_cache(
        candidates_by_depth=heap,
        node_state_by_id=states,
        candidate_versions_by_node_id=versions,
    )
    second = candidate_payloads_from_cache(
        candidates_by_depth={2: [(-9.0, 10, 13)]},
        node_state_by_id=states,
        candidate_versions_by_node_id={10: 13},
    )
    assert first == second
    assert len(first[0].candidates) == 1
    assert (
        first[0].candidates[0].priority == 9.0 and first[0].candidates[0].version == 1
    )
    assert heap == {2: [(-9.0, 10, 7), (-100.0, 10, 6)]} and versions == {10: 7}


@pytest.mark.parametrize("storage", ["typed", "json-zst", "sharded", "split"])
def test_exact_and_estimated_score_ties_keep_independent_certainties(
    storage: str, tmp_path: Path
) -> None:
    from valanga.evaluations import Certainty, Value

    from tests.test_checkpoint_load import _FakeIncrementalStateCheckpointCodec

    graph = {0: [1, 2], 1: [], 2: [3], 3: []}
    search = runtime(graph, {0: 0.0, 1: 9.0, 2: 1.0, 3: 0.0}, steps=1)
    terminal = next(n for n in nodes(search) if n.tag == 1)
    estimated = next(n for n in nodes(search) if n.tag == 2)
    estimated.tree_evaluation.direct_value = Value(
        score=9.0, certainty=Certainty.ESTIMATE
    )
    search.tree_manager.value_propagator.propagate_from_changed_nodes([estimated])
    assert terminal.tree_evaluation.direct_value.certainty == Certainty.TERMINAL
    assert (
        search.tree.root_node.tree_evaluation.tree_value.certainty == Certainty.ESTIMATE
    )
    assert snapshot(search) == reference(search)
    codec = _FakeIncrementalStateCheckpointCodec(graph)
    restored = roundtrip(search, codec, storage, tmp_path)
    assert ordered(restored) == ordered(search)
    assert snapshot(restored) == snapshot(search) == reference(restored)
