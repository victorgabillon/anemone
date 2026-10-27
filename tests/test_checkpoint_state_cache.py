"""Bounded decoded retention, payload authority and lazy-handle correctness."""
# ruff: noqa: D103

from itertools import pairwise
from random import Random
from typing import Any

import pytest
from valanga import Color

from anemone import create_search_with_tree_eval_factory
from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    CheckpointBackedStateHandle,
    CheckpointStateResolver,
    DeltaCheckpointStatePayload,
    build_search_checkpoint_payload,
    load_search_from_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    serialize_checkpoint_atom,
    write_sharded_search_checkpoint,
)
from anemone.checkpoints.state_handles import CheckpointStateResolutionError
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from tests.fake_yaml_game import FakeYamlDynamics, MasterStateValueEvaluatorFromYaml
from tests.scientific_refresh_helpers import nodes, reference, snapshot
from tests.test_checkpoint_load import (
    _build_linoo_args,
    _ConcreteFakeYamlState,
    _FakeIncrementalStateCheckpointCodec,
)


def anchor(node_id: int) -> AnchorCheckpointStatePayload:
    """Create an independently reconstructible fixture anchor."""
    return AnchorCheckpointStatePayload(
        anchor_ref={"node_id": node_id, "turn": "WHITE"}
    )


def delta(node_id: int, parent_id: int) -> DeltaCheckpointStatePayload:
    """Use one parent reference and a checked domain delta."""
    return DeltaCheckpointStatePayload(
        state_parent_node_id=parent_id,
        state_parent_branch=serialize_checkpoint_atom(7),
        delta_ref={
            "parent_node_id": parent_id,
            "child_node_id": node_id,
            "branch_from_parent": 7,
            "turn": "WHITE",
        },
    )


@pytest.mark.parametrize("capacity", [0, 1, 3, 256, 1024])
def test_long_sparse_chain_and_repeated_resolution_after_eviction(
    capacity: int,
) -> None:
    ids = [11 + 7 * i for i in range(180)]
    payloads = {ids[0]: anchor(ids[0])} | {
        child: delta(child, parent) for parent, child in pairwise(ids)
    }
    codec = _FakeIncrementalStateCheckpointCodec({})
    resolver = CheckpointStateResolver(
        state_codec=codec,
        state_payloads_by_node_id=payloads,
        state_cache_capacity=capacity,
    )
    handle = CheckpointBackedStateHandle(resolver, ids[-1])
    first = handle.get()
    assert first.node_id == ids[-1]
    assert len(resolver._resolved_states) <= capacity
    for node_id in reversed(ids):
        state = resolver.resolve(node_id)
        assert state.node_id == node_id and state.turn == Color.WHITE
        assert len(resolver._resolved_states) <= capacity
        assert not resolver._resolving_node_ids
        assert resolver.payload_for_node_id(node_id) is payloads[node_id]
    assert handle.get() == first
    assert first.node_id == ids[-1]  # Eviction never invalidates external references.
    assert handle.checkpoint_payload_for_reuse_or_none() is payloads[ids[-1]]


def test_lru_hits_protect_recently_used_states() -> None:
    codec = _FakeIncrementalStateCheckpointCodec({})
    resolver = CheckpointStateResolver(
        codec, {i: anchor(i) for i in (10, 20, 30)}, state_cache_capacity=2
    )
    first = resolver.resolve(10)
    resolver.resolve(20)
    assert resolver.resolve(10) is first
    resolver.resolve(30)
    assert tuple(resolver._resolved_states) == (10, 30)
    resolver.resolve(20)
    assert codec.loaded_anchor_node_ids == [10, 20, 30, 20]
    assert tuple(resolver._resolved_states) == (30, 20)


@pytest.mark.parametrize("capacity", [0, 1, 2])
def test_active_parent_remains_alive_during_reentrant_eviction(capacity: int) -> None:
    class Codec(_FakeIncrementalStateCheckpointCodec):
        """Force eviction inside the existing child decoder while parent is live."""

        def load_child_from_delta(
            self,
            *,
            parent_state: Any,
            delta_ref: object,
            branch_from_parent: object = None,
        ) -> Any:
            before = parent_state.node_id
            resolver.resolve(99)
            assert parent_state.node_id == before
            return super().load_child_from_delta(
                parent_state=parent_state,
                delta_ref=delta_ref,
                branch_from_parent=branch_from_parent,
            )

    codec = Codec({})
    resolver = CheckpointStateResolver(
        codec,
        {10: anchor(10), 20: delta(20, 10), 30: delta(30, 20), 99: anchor(99)},
        state_cache_capacity=capacity,
    )
    assert resolver.resolve(30).node_id == 30
    assert len(resolver._resolved_states) <= capacity
    assert not resolver._resolving_node_ids


@pytest.mark.parametrize("capacity", [0, 1, 4])
def test_shared_anchor_and_sibling_delta_chains(capacity: int) -> None:
    codec = _FakeIncrementalStateCheckpointCodec({})
    payloads = {
        10: anchor(10),
        20: delta(20, 10),
        30: delta(30, 10),
        40: delta(40, 20),
        50: delta(50, 30),
    }
    resolver = CheckpointStateResolver(codec, payloads, state_cache_capacity=capacity)
    for node_id in (40, 50, 20, 30, 40, 50):
        assert resolver.resolve(node_id).node_id == node_id
        assert len(resolver._resolved_states) <= capacity
    assert payloads[40].state_parent_node_id == 20
    assert resolver.payload_for_node_id(40) is payloads[40]


@pytest.mark.parametrize("capacity", [0, 1, 8])
@pytest.mark.parametrize("failure", ["missing", "self", "cycle", "anchor"])
def test_errors_keep_cycle_detection_and_release_active_chain(
    capacity: int, failure: str
) -> None:
    payloads = {10: anchor(10), 20: delta(20, 10)}
    if failure == "missing":
        payloads[20] = delta(20, 12345)
    elif failure == "self":
        payloads[20] = delta(20, 20)
    elif failure == "cycle":
        payloads[20], payloads[30] = delta(20, 30), delta(30, 20)
    else:
        payloads[20] = AnchorCheckpointStatePayload(anchor_ref={})
    resolver = CheckpointStateResolver(
        _FakeIncrementalStateCheckpointCodec({}),
        payloads,
        state_cache_capacity=capacity,
    )
    error = CheckpointStateResolutionError if failure in {"self", "cycle"} else KeyError
    match = (
        "Cycle" if failure == "cycle" else "no parent" if failure == "self" else None
    )
    with pytest.raises(error, match=match):
        resolver.resolve(20)
    assert not resolver._resolving_node_ids
    assert resolver.resolve(10).node_id == 10
    assert len(resolver._resolved_states) <= capacity


def test_invalid_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="state_cache_capacity must be non-negative"):
        CheckpointStateResolver(
            _FakeIncrementalStateCheckpointCodec({}), {}, state_cache_capacity=-1
        )


@pytest.mark.parametrize("capacity", [0, 1, 4, 256, 1024])
@pytest.mark.parametrize("storage", ["typed", "sharded", "split"])
def test_restore_refresh_and_growth_preserve_values_dag_rng_and_selector(
    capacity: int, storage: str, tmp_path: Any
) -> None:
    children = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: [5], 5: []}
    values = {i: float(i) for i in children}
    args = _build_linoo_args()
    args.node_selector.base.depth_selection_policy = "alternating_by_step"
    evaluator = MasterStateValueEvaluatorFromYaml(values)
    continuous = create_search_with_tree_eval_factory(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        starting_state=_ConcreteFakeYamlState(0, children),
        args=args,
        random_generator=Random(0),
        master_state_evaluator=evaluator,
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    for _ in range(3):
        continuous.step()
    codec = _FakeIncrementalStateCheckpointCodec(children)
    payload = build_search_checkpoint_payload(continuous, state_codec=codec)
    args = _build_linoo_args()
    args.node_selector.base.depth_selection_policy = "alternating_by_step"
    kwargs = dict(
        state_codec=codec,
        dynamics=FakeYamlDynamics(),
        args=args,
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=evaluator,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        state_cache_capacity=capacity,
    )
    if storage == "typed":
        restored = load_search_from_checkpoint_payload(payload, **kwargs)
    else:
        write_sharded_search_checkpoint(
            payload,
            tmp_path,
            node_count_per_shard=2,
            layout="split" if storage == "split" else "node_records",
        )
        restored = load_search_from_sharded_checkpoint(tmp_path, **kwargs)
    resolver = restored.tree.root_node.tree_node.state_handle.resolver
    assert resolver.state_cache_capacity == capacity
    assert snapshot(restored) == snapshot(continuous)
    assert any(len(n.parent_nodes) > 1 for n in nodes(restored))
    for _ in range(2):
        expected = reference(restored, evaluator)
        restored.refresh_with_evaluator(evaluator, scope="all", batch_size=1)
        continuous.refresh_with_evaluator(evaluator, scope="all", batch_size=1)
        assert snapshot(restored) == expected == snapshot(continuous)
        assert len(resolver._resolved_states) <= capacity
    for _ in range(3):
        continuous.step()
        restored.step()
        assert (
            restored.latest_step_report.selected_node_id
            == continuous.latest_step_report.selected_node_id
        )
        assert snapshot(restored) == snapshot(continuous)
        assert (
            restored.node_selector.base.random_generator.getstate()
            == continuous.node_selector.base.random_generator.getstate()
        )
        assert (
            restored.node_selector.base._selection_step_count
            == continuous.node_selector.base._selection_step_count
        )
        assert len(resolver._resolved_states) <= capacity


def test_default_capacity_bounds_a_complete_population_scan() -> None:
    resolver = CheckpointStateResolver(
        _FakeIncrementalStateCheckpointCodec({}), {i: anchor(i) for i in range(2048)}
    )
    for node_id in range(2048):
        assert resolver.resolve(node_id).node_id == node_id
        assert len(resolver._resolved_states) <= 256


@pytest.mark.parametrize("capacity", [0, 1, 7, 1024])
def test_real_seven_node_morpion_refresh_and_growth_after_eviction(
    capacity: int,
) -> None:
    from tests.morpion_checkpoint_cache_helpers import Codec
    from tests.morpion_refresh_helpers import Dynamics, Evaluator, State, runtime

    continuous = runtime(branch_limit=100000)
    codec = Codec()
    payload = build_search_checkpoint_payload(continuous, state_codec=codec)
    args = _build_linoo_args()
    args.stopping_criterion.tree_branch_limit = 100000
    args.node_selector.base.depth_selection_policy = "alternating_by_step"
    restored = load_search_from_checkpoint_payload(
        payload,
        state_codec=codec,
        dynamics=Dynamics(),
        args=args,
        state_type=State,
        master_state_value_evaluator=Evaluator(),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        state_cache_capacity=capacity,
    )
    resolver = restored.tree.root_node.tree_node.state_handle.resolver
    before_states = {n.id: n.state.board.canonical_tag for n in nodes(continuous)}
    assert {n.id: n.state.board.canonical_tag for n in nodes(restored)} == before_states
    for score in (42.0, 7.0, 42.0):
        evaluator = Evaluator({n.id: score for n in nodes(restored)})
        expected = reference(restored, evaluator)
        restored.refresh_with_evaluator(evaluator, scope="all", batch_size=1)
        continuous.refresh_with_evaluator(evaluator, scope="all", batch_size=1)
        assert snapshot(restored) == expected == snapshot(continuous)
        assert len(resolver._resolved_states) <= capacity
    for _ in range(3):
        continuous.step()
        restored.step()
        assert (
            restored.latest_step_report.selected_node_id
            == continuous.latest_step_report.selected_node_id
        )
        assert snapshot(restored) == snapshot(continuous)
        assert (
            restored.node_selector.base.random_generator.getstate()
            == continuous.node_selector.base.random_generator.getstate()
        )
        assert len(resolver._resolved_states) <= capacity
