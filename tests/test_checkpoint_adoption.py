"""Committed live graph adoption, with exact cold-reference checkpoint parity."""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import pytest

from anemone.checkpoints import (
    CheckpointAdoptionError,
    DeltaCheckpointStatePayload,
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
    prepare_checkpoint_adoption,
)
from anemone.nodes.state_handles import MaterializedStateHandle
from tests.test_checkpoint_load import (
    _build_linoo_runtime,
    _build_runtime,
    _FakeIncrementalStateCheckpointCodec,
    _load_runtime_from_payload,
    _SelectiveDeltaCheckpointCodec,
)

GRAPH = {0: [1, 2, 6], 1: [3, 3], 2: [3], 3: [4], 4: [5], 5: [], 6: [7], 7: []}


def _join(runtime: Any) -> Any:
    return next(
        node for node in runtime._all_nodes_in_tree_order() if node.parent_count() > 1
    )


def _fixture(kind: str) -> tuple[Any, Any, Any]:
    runtime = _build_runtime(children_by_id=GRAPH)
    runtime.step()
    runtime.step()
    node = _join(runtime)
    order = sorted(
        (parent for parent, _ in node.iter_parent_items()),
        key=lambda parent: parent.id,
        reverse=True,
    )
    node.tree_node.reorder_parent_links(order)
    rejected = {(1, 3), (2, 3)} if kind == "anchor" else set()
    if kind == "alternate_delta":
        rejected = {(order[0].state.node_id, 3)}
    codec = _SelectiveDeltaCheckpointCodec(GRAPH, rejected)
    payload = build_search_checkpoint_payload(runtime, state_codec=codec)
    return runtime, codec, payload


def _snapshot(runtime: Any, codec: Any) -> Any:
    return checkpoint_payload_to_jsonable(
        build_search_checkpoint_payload(runtime, state_codec=codec)
    )


@pytest.mark.parametrize("kind", ["anchor", "delta", "alternate_delta"])
def test_adoption_matches_full_cold_checkpoint_and_repeated_continuation(
    kind: str,
) -> None:
    """Reproduce both distinct root causes, keeping every serialized field exact."""
    warm, codec, payload = _fixture(kind)
    cold = _load_runtime_from_payload(payload, state_codec=codec, children_by_id=GRAPH)
    if kind != "delta":
        # Failing-before evidence: unchanged graph, different serialized parent.
        assert _snapshot(warm, codec) != _snapshot(cold, codec)
    assert all(
        isinstance(node.state_handle, MaterializedStateHandle)
        for node in warm._all_nodes_in_tree_order()
    )
    identity = id(warm)
    nodes = tuple(warm._all_nodes_in_tree_order())
    for _ in range(3):
        plan = prepare_checkpoint_adoption(warm, payload, state_codec=codec)
        stats = plan.adopt()
        assert stats.nodes_bound == len(payload.tree.nodes)
        assert id(warm) == identity
        assert tuple(warm._all_nodes_in_tree_order())[: len(nodes)] == nodes
        assert _snapshot(warm, codec) == _snapshot(cold, codec)
        warm.step()
        cold.step()
        assert _snapshot(warm, codec) == _snapshot(cold, codec)
        payload = build_search_checkpoint_payload(warm, state_codec=codec)
        cold = _load_runtime_from_payload(
            payload, state_codec=codec, children_by_id=GRAPH
        )


def test_adopted_save_restore_save_is_idempotent() -> None:
    """Preserve the historical cold policy and stabilize at its existing boundary."""
    runtime, codec, original = _fixture("anchor")
    prepare_checkpoint_adoption(runtime, original, state_codec=codec).adopt()
    saved = build_search_checkpoint_payload(runtime, state_codec=codec)
    cold = _load_runtime_from_payload(saved, state_codec=codec, children_by_id=GRAPH)
    assert _snapshot(runtime, codec) == _snapshot(cold, codec)
    # No format or legacy first-save policy was changed to make this pass.
    assert saved.format_version == original.format_version == 2


def test_adoption_does_not_decode_and_preserves_external_handles() -> None:
    """Keep resolver/cache ownership valid while binding new lazy handles."""
    runtime, codec, payload = _fixture("anchor")
    prepare_checkpoint_adoption(runtime, payload, state_codec=codec).adopt()
    node = _join(runtime)
    old_handle = node.state_handle
    expected = old_handle.get()
    old_resolver = old_handle.resolver
    old_cache = old_resolver._resolved_states
    old_items = dict(old_cache)
    codec.reset_counters()
    stats = prepare_checkpoint_adoption(runtime, payload, state_codec=codec).adopt()
    assert stats.nodes_bound == 6
    assert codec.loaded_anchor_node_ids == []
    assert codec.applied_delta_items == []
    assert old_resolver._resolved_states is old_cache
    assert old_cache == old_items
    assert old_handle.get() is expected
    assert node.state_handle.resolver is not old_resolver
    assert node.state_handle.resolver._resolved_states == {}
    assert node.state.tag == expected.tag


def test_multiple_branches_and_parent_dictionary_owners_survive_reordering() -> None:
    """Reorder navigation only; retain exact edge sets, mapping and branch owners."""
    runtime, codec, payload = _fixture("anchor")
    node = _join(runtime)
    mapping = node.parent_nodes
    view = node.parent_nodes_view()
    branches = dict(mapping)
    assert sorted(len(value) for value in branches.values()) == [1, 2]
    prepare_checkpoint_adoption(runtime, payload, state_codec=codec).adopt()
    assert node.parent_nodes is mapping
    assert dict(view) == mapping == branches
    assert all(mapping[parent] is value for parent, value in branches.items())
    assert [parent.id for parent in view] == sorted(parent.id for parent in view)


@pytest.mark.parametrize("kind", ["anchor", "delta", "alternate_delta"])
def test_saved_ancestry_decodes_to_the_same_states_without_replay_storms(
    kind: str,
) -> None:
    """Adopted opaque deltas remain codec-valid, shared and memoized."""
    runtime, codec, payload = _fixture(kind)
    expected = {node.id: node.state.tag for node in runtime._all_nodes_in_tree_order()}
    codec.reset_counters()
    prepare_checkpoint_adoption(runtime, payload, state_codec=codec).adopt()
    assert codec.loaded_anchor_node_ids == codec.applied_delta_items == []
    for node in runtime._all_nodes_in_tree_order():
        assert node.state.tag == expected[node.id]
    loads = len(codec.loaded_anchor_node_ids) + len(codec.applied_delta_items)
    for node in runtime._all_nodes_in_tree_order():
        assert node.state.tag == expected[node.id]
    assert (
        len(codec.loaded_anchor_node_ids) + len(codec.applied_delta_items)
        == loads
        == len(expected)
    )


@pytest.mark.parametrize(
    "mutation", ["growth", "rng", "evaluator", "parent_edge", "opening"]
)
def test_stale_plan_refuses_before_modifying_handles(mutation: str) -> None:
    """Refuse supported mutation instead of silently rebinding stale persistence."""
    runtime, codec, payload = _fixture("anchor")
    if mutation == "rng":
        runtime = _build_linoo_runtime()
        runtime.step()
        codec = _FakeIncrementalStateCheckpointCodec(
            {0: [1, 2], 1: [3], 2: [4], 3: [], 4: []}
        )
        payload = build_search_checkpoint_payload(runtime, state_codec=codec)
    plan = prepare_checkpoint_adoption(runtime, payload, state_codec=codec)
    if mutation == "growth":
        runtime.step()
    elif mutation == "rng":
        runtime.node_selector.base.random_generator.random()
    elif mutation == "evaluator":
        runtime.evaluator_version += 1
    elif mutation == "parent_edge":
        _join(runtime).tree_node.parent_nodes.pop(
            next(iter(_join(runtime).parent_nodes))
        )
    else:
        _join(runtime).all_branches_generated = not _join(
            runtime
        ).all_branches_generated
    handles = [node.state_handle for node in runtime._all_nodes_in_tree_order()]
    with pytest.raises(CheckpointAdoptionError):
        plan.adopt()
    assert [node.state_handle for node in runtime._all_nodes_in_tree_order()] == handles


def test_plan_is_single_use_and_does_not_retain_runtime_graph() -> None:
    """An abandoned plan never owns the runtime or replaces a handle implicitly."""
    runtime, codec, payload = _fixture("anchor")
    plan = prepare_checkpoint_adoption(runtime, payload, state_codec=codec)
    plan.adopt()
    with pytest.raises(CheckpointAdoptionError, match="plan_already_consumed"):
        plan.adopt()
    runtime_ref = weakref.ref(runtime)
    del runtime
    gc.collect()
    assert runtime_ref() is None


def test_interrupted_write_discards_plan_without_mutation() -> None:
    """No prepare/save side effect commits references before publication succeeds."""
    runtime, codec, payload = _fixture("anchor")
    nodes = tuple(runtime._all_nodes_in_tree_order())
    handles = [node.state_handle for node in nodes]
    parents = [list(node.iter_parent_items()) for node in nodes]
    plan = prepare_checkpoint_adoption(runtime, payload, state_codec=codec)
    writer = Mock(side_effect=OSError("interrupted_checkpoint_write"))
    with pytest.raises(OSError, match="interrupted_checkpoint_write"):
        writer()
    del plan
    assert [node.state_handle for node in nodes] == handles
    assert [list(node.iter_parent_items()) for node in nodes] == parents
    assert _snapshot(runtime, codec) == checkpoint_payload_to_jsonable(payload)


def test_cyclic_delta_ancestry_is_rejected_without_decodes() -> None:
    """Validate the compact reference graph before giving it to lazy handles."""
    runtime, codec, payload = _fixture("delta")
    a, b = payload.tree.nodes[1:3]
    bad = replace(
        payload,
        tree=replace(
            payload.tree,
            nodes=[
                replace(
                    node,
                    state_payload=DeltaCheckpointStatePayload(
                        state_parent_node_id=b.node_id
                        if node.node_id == a.node_id
                        else a.node_id,
                        delta_ref={},
                        state_parent_branch=None,
                    ),
                )
                if node.node_id in [a.node_id, b.node_id]
                else node
                for node in payload.tree.nodes
            ],
        ),
    )
    codec.reset_counters()
    with pytest.raises(CheckpointAdoptionError, match="cyclic_state_ancestry"):
        prepare_checkpoint_adoption(runtime, bad, state_codec=codec)
    assert codec.loaded_anchor_node_ids == codec.applied_delta_items == []


def test_linoo_selector_rng_values_and_object_identity_are_untouched() -> None:
    """Adopt state/navigation ownership without replacing unrelated runtime policy."""
    runtime = _build_linoo_runtime()
    runtime.step()
    codec = _FakeIncrementalStateCheckpointCodec(
        {0: [1, 2], 1: [3], 2: [4], 3: [], 4: []}
    )
    payload = build_search_checkpoint_payload(runtime, state_codec=codec)
    before = _snapshot(runtime, codec)
    selector = runtime.node_selector
    evaluation = runtime.tree.root_node.tree_evaluation
    prepare_checkpoint_adoption(runtime, payload, state_codec=codec).adopt()
    assert runtime.node_selector is selector
    assert runtime.tree.root_node.tree_evaluation is evaluation
    assert _snapshot(runtime, codec) == before


def test_reorder_parent_links_rejects_membership_changes() -> None:
    """The navigation API may reorder but cannot lose or duplicate an edge."""
    runtime, _, _ = _fixture("anchor")
    node = _join(runtime)
    parents = list(node.parent_nodes)
    original = list(node.iter_parent_items())
    with pytest.raises(ValueError, match="invalid_parent_order"):
        node.tree_node.reorder_parent_links([parents[0], parents[0]])
    assert list(node.iter_parent_items()) == original


@pytest.mark.parametrize(
    "random_kind", ["random_openable", "random_legal_prefer_openable"]
)
def test_independent_rollout_rng_is_preserved_and_drift_refuses(
    random_kind: str,
) -> None:
    """Never draw/reset rollout RNG or decode state while adopting its graph."""
    from anemone.checkpoints.rng_state import rollout_random_generator
    from anemone.tree_manager.opening_expansion_config import RolloutActionSelectorKind
    from tests.test_checkpoint_rollout_rng import _GRAPH, _args, _runtime

    runtime = _runtime(_args(RolloutActionSelectorKind(random_kind)))
    runtime.step()
    codec = _FakeIncrementalStateCheckpointCodec(_GRAPH)
    payload = build_search_checkpoint_payload(runtime, state_codec=codec)
    before = _snapshot(runtime, codec)
    codec.reset_counters()
    prepare_checkpoint_adoption(runtime, payload, state_codec=codec).adopt()
    assert codec.loaded_anchor_node_ids == codec.applied_delta_items == []
    assert _snapshot(runtime, codec) == before
    plan = prepare_checkpoint_adoption(runtime, payload, state_codec=codec)
    rng = rollout_random_generator(runtime)
    assert rng is not None
    rng.random()
    with pytest.raises(CheckpointAdoptionError, match="runtime_changed_since_prepare"):
        plan.adopt()


def test_reevaluation_value_updates_are_never_rolled_back() -> None:
    """The state-reference lifecycle does not replace value/evaluator policy."""
    from anemone.value_updates import NodeValueUpdate

    runtime = _build_linoo_runtime()
    runtime.step()
    codec = _FakeIncrementalStateCheckpointCodec(
        {0: [1, 2], 1: [3], 2: [4], 3: [], 4: []}
    )
    payload = build_search_checkpoint_payload(runtime, state_codec=codec)
    plan = prepare_checkpoint_adoption(runtime, payload, state_codec=codec)
    runtime.apply_node_value_updates(
        [NodeValueUpdate(node_id="1", direct_value=123.0, is_exact=False)]
    )
    updated = _snapshot(runtime, codec)
    plan.adopt()
    assert _snapshot(runtime, codec) == updated


def test_old_json_payload_without_summaries_keeps_lazy_adoption_compatible() -> None:
    """Existing format-2 checkpoint fields remain readable without a new schema."""
    from tests.test_checkpoint_rollout_rng import _decode_json

    runtime, codec, payload = _fixture("anchor")
    raw = checkpoint_payload_to_jsonable(payload)
    raw.pop("rng_state", None)
    raw.pop("rollout_rng_state", None)
    for node in raw["tree"]["nodes"]:
        node["state_payload"].pop("state_summary", None)
    legacy = _decode_json(raw)
    codec.reset_counters()
    prepare_checkpoint_adoption(runtime, legacy, state_codec=codec).adopt()
    assert codec.loaded_anchor_node_ids == codec.applied_delta_items == []
    cold = _load_runtime_from_payload(legacy, state_codec=codec, children_by_id=GRAPH)
    assert _snapshot(runtime, codec) == _snapshot(cold, codec)


def test_second_valid_branch_from_same_parent_keeps_exact_reconstruction_ancestry() -> (
    None
):
    """Saved branch identity matters even when the representative parent is equal."""

    class BranchCodec(_FakeIncrementalStateCheckpointCodec):
        def dump_delta_from_parent(
            self,
            *,
            parent_state: Any,
            child_state: Any,
            branch_from_parent: object | None = None,
        ) -> object:
            if (
                parent_state.node_id == 1
                and child_state.node_id == 3
                and branch_from_parent == 0
            ):
                raise ValueError("first_branch_rejected")
            return super().dump_delta_from_parent(
                parent_state=parent_state,
                child_state=child_state,
                branch_from_parent=branch_from_parent,
            )

    runtime, _, _ = _fixture("delta")
    node = _join(runtime)
    preferred = next(
        parent for parent, branches in node.iter_parent_items() if len(branches) == 2
    )
    node.tree_node.reorder_parent_links(
        [
            preferred,
            *(
                parent
                for parent, _ in node.iter_parent_items()
                if parent is not preferred
            ),
        ]
    )
    codec = BranchCodec(GRAPH)
    saved = build_search_checkpoint_payload(runtime, state_codec=codec)
    cold = _load_runtime_from_payload(saved, state_codec=codec, children_by_id=GRAPH)
    assert _snapshot(runtime, codec) != _snapshot(cold, codec)
    prepare_checkpoint_adoption(runtime, saved, state_codec=codec).adopt()
    assert _snapshot(runtime, codec) == _snapshot(cold, codec)
    assert node.state.tag == cold.tree.descendants[2][node.state.tag].state.tag
