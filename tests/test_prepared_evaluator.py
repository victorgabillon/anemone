"""Real Morpion parity and fail-closed native preparation/installation tests."""

# ruff: noqa: D103
from dataclasses import replace
from random import Random
from time import monotonic
from typing import Any

import pytest
from atomheart.games.morpion import initial_state

from anemone import create_search_with_tree_eval_factory
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    load_search_from_checkpoint_payload,
)
from anemone.indices.node_indices.index_types import IndexComputationType
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.prepared_evaluator import (
    EvaluationIdentity,
    OwnedInlineCodecCapabilities,
    PreparationLimits,
    SourceIdentity,
    SupportedSearchRecipe,
    freeze_evaluator_source,
    prepare_evaluator_context,
)
from anemone.prepared_evaluator.native import derived, encode
from tests.morpion_checkpoint_cache_helpers import Codec as BaseCodec
from tests.morpion_refresh_helpers import Dynamics, Evaluator, State
from tests.test_checkpoint_load import _build_linoo_args

IDENTITY = EvaluationIdentity(
    "bundle", "weights", "architecture", "encoder", "morpion-input-v1", "max"
)


class Codec(BaseCodec):
    """Explicitly admit the existing fresh inline Morpion test adapter."""

    prepared_capabilities = OwnedInlineCodecCapabilities("morpion-inline-v1")


class Incumbent(Evaluator):
    """Bind the incumbent model identity for exact staging admission."""

    evaluation_identity = "incumbent"


class Candidate(Evaluator):
    """Bound deterministic native scalar evaluator identity."""

    evaluation_identity = IDENTITY


def limits() -> PreparationLimits:
    return PreparationLimits(monotonic() + 30)


INDEX_RECIPES = [None, IndexComputationType.MIN_GLOBAL_CHANGE]


def args(
    index_computation: IndexComputationType
    | None = IndexComputationType.MIN_GLOBAL_CHANGE,
) -> Any:
    result = _build_linoo_args()
    result.index_computation = index_computation
    result.stopping_criterion.tree_branch_limit = 4096
    result.node_selector.base.depth_selection_policy = "alternating_by_step"
    return result


def runtime(
    steps: int = 3,
    index_computation: IndexComputationType
    | None = IndexComputationType.MIN_GLOBAL_CHANGE,
) -> Any:
    search = create_search_with_tree_eval_factory(
        state_type=State,
        dynamics=Dynamics(),
        starting_state=State(initial_state()),
        args=args(index_computation),
        random_generator=Random(7),
        master_state_evaluator=Evaluator(),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    for _ in range(steps):
        search.step()
    return search


def cold(
    payload: Any,
    index_computation: IndexComputationType
    | None = IndexComputationType.MIN_GLOBAL_CHANGE,
) -> Any:
    return load_search_from_checkpoint_payload(
        payload,
        state_codec=Codec(),
        dynamics=Dynamics(),
        args=args(index_computation),
        state_type=State,
        master_state_value_evaluator=Incumbent(),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )


def freeze(
    payload: Any,
    index_computation: Any = IndexComputationType.MIN_GLOBAL_CHANGE,
) -> Any:
    return freeze_evaluator_source(
        payload,
        codec_type=Codec,
        identity=SourceIdentity(
            "publication", "incumbent", "morpion-inline-v1", "test-build", "morpion"
        ),
        recipe=SupportedSearchRecipe(index_computation=index_computation),
        limits=limits(),
    )


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_real_morpion_detached_refresh_matches_cold_live(
    index_computation: Any,
) -> None:
    payload = build_search_checkpoint_payload(
        runtime(index_computation=index_computation), state_codec=Codec()
    )
    source = freeze(payload, index_computation)
    original = source._checkpoint_bytes
    candidate = Candidate({i: 42.0 + i for i in range(512)})
    context = prepare_evaluator_context(
        source, evaluator=candidate, evaluation_identity=IDENTITY, limits=limits()
    )
    reference = cold(payload, index_computation)
    expected = reference.refresh_with_evaluator(
        Candidate({i: 42.0 + i for i in range(512)}), scope="all"
    )
    assert context._derived_bytes == encode(
        derived(reference.tree, reference.node_selector.base)
    )
    assert context.requested_count == expected.requested_count
    assert context.evaluator_version == reference.evaluator_version
    assert source._checkpoint_bytes == original
    assert not hasattr(context, "to_checkpoint_payload")


def test_source_is_immutable_and_bad_identity_refuses() -> None:
    payload = build_search_checkpoint_payload(runtime(), state_codec=Codec())
    source = freeze(payload)
    payload.tree.nodes.clear()
    assert source._checkpoint_bytes
    with pytest.raises(ValueError, match="model identity"):
        prepare_evaluator_context(
            source,
            evaluator=Candidate(),
            evaluation_identity=replace(IDENTITY, weights="other"),
            limits=limits(),
        )


@pytest.mark.parametrize(
    "fault", ["missing-parent", "cycle", "index", "selector", "codec", "limit"]
)
@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_source_refuses_incomplete_or_unsupported_input(
    fault: str, index_computation: Any
) -> None:
    from anemone.checkpoints.payloads import ExplorationIndexCheckpointPayload

    payload = build_search_checkpoint_payload(
        runtime(index_computation=index_computation), state_codec=Codec()
    )
    if fault == "missing-parent":
        payload.tree.nodes[1].parent_node_id = 999999
    elif fault == "cycle":
        node = payload.tree.nodes[1]
        node.state_payload = replace(
            node.state_payload, state_parent_node_id=node.node_id
        )
    elif fault == "index":
        payload.tree.nodes[1].exploration_index = (
            ExplorationIndexCheckpointPayload(kind="MinMaxPathValue", payload={})
            if index_computation is None
            else None
        )
    elif fault == "selector":
        payload.selector_state = None
    with pytest.raises(ValueError):
        freeze_evaluator_source(
            payload,
            codec_type=BaseCodec if fault == "codec" else Codec,
            identity=SourceIdentity("p", "i", "c", "b", "test"),
            recipe=SupportedSearchRecipe(index_computation=index_computation),
            limits=replace(limits(), max_nodes=1) if fault == "limit" else limits(),
        )


def test_detached_structural_mutation_is_unavailable() -> None:
    class MutatingCandidate(Candidate):
        def evaluate_batch_items(self, items: Any) -> Any:
            items[0].clear_child_links()
            return super().evaluate_batch_items(items)

    source = freeze(build_search_checkpoint_payload(runtime(), state_codec=Codec()))
    with pytest.raises(TypeError, match="read-only"):
        prepare_evaluator_context(
            source,
            evaluator=MutatingCandidate(),
            evaluation_identity=IDENTITY,
            limits=limits(),
        )


def prepared_case(
    index_computation: IndexComputationType
    | None = IndexComputationType.MIN_GLOBAL_CHANGE,
) -> Any:
    from anemone.prepared_evaluator.native import decode

    source = freeze(
        build_search_checkpoint_payload(
            runtime(index_computation=index_computation), state_codec=Codec()
        ),
        index_computation,
    )
    candidate = Candidate({i: float(i + 42) for i in range(512)})
    prepared = prepare_evaluator_context(
        source, evaluator=candidate, evaluation_identity=IDENTITY, limits=limits()
    )
    return (
        source,
        prepared,
        candidate,
        cold(decode(source._checkpoint_bytes), index_computation),
    )


def stage(source: Any, prepared: Any, candidate: Any, owner: Any) -> Any:
    from anemone.prepared_evaluator import stage_evaluator_context

    return stage_evaluator_context(
        owner,
        current_source=source,
        prepared=prepared,
        evaluator=candidate,
        evaluation_identity=IDENTITY,
        limits=limits(),
    )


def native_bytes(search: Any) -> bytes:
    from anemone.checkpoints.io import _encode_checkpoint_payload

    return _encode_checkpoint_payload(
        build_search_checkpoint_payload(search, state_codec=Codec())
    ).json_bytes


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_real_install_cold_export_and_next_growth(index_computation: Any) -> None:
    from anemone.prepared_evaluator.native import decode

    source, prepared, candidate, owner = prepared_case(index_computation)
    reference = cold(decode(source._checkpoint_bytes), index_computation)
    reference.refresh_with_evaluator(
        Candidate({i: float(i + 42) for i in range(512)}), scope="all"
    )
    nodes = list(owner._iter_nodes_by_depth())
    handles = [n.state_handle for n in nodes]
    rng = owner.node_selector.base.random_generator
    state = rng.getstate()
    plan = stage(source, prepared, candidate, owner)
    report = plan.install()
    assert report.installed_nodes == len(nodes)
    assert owner.node_selector.base.random_generator is rng and rng.getstate() == state
    assert [n.state_handle for n in nodes] == handles
    assert native_bytes(owner) == native_bytes(reference)
    for _ in range(2):
        before = owner.tree.branch_count
        owner.step()
        reference.step()
        assert owner.tree.branch_count > before
        assert native_bytes(owner) == native_bytes(reference)
    with pytest.raises(ValueError, match="consumed"):
        plan.install()


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_stale_preflight_preserves_owner(index_computation: Any) -> None:
    source, prepared, candidate, owner = prepared_case(index_computation)
    plan = stage(source, prepared, candidate, owner)
    owner.node_selector.base.random_generator.random()
    before = native_bytes(owner)
    with pytest.raises(ValueError, match="stale head"):
        plan.install()
    assert native_bytes(owner) == before and not owner._prepared_evaluator_invalidated


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_failure_after_mutation_invalidates_runtime(
    monkeypatch: Any, index_computation: Any
) -> None:
    from anemone.prepared_evaluator import RuntimeInvalidated, installation

    source, prepared, candidate, owner = prepared_case(index_computation)
    plan = stage(source, prepared, candidate, owner)
    assign = installation._assign_node

    def fail(node: Any, staged: Any) -> None:
        assign(node, staged)
        message = "injected after first assignment"
        raise RuntimeError(message)

    monkeypatch.setattr(installation, "_assign_node", fail)
    with pytest.raises(RuntimeInvalidated):
        plan.install()
    assert owner._prepared_evaluator_invalidated
    for operation in (
        owner.step,
        lambda: native_bytes(owner),
        lambda: owner.refresh_with_evaluator(candidate, scope="all"),
    ):
        with pytest.raises(RuntimeInvalidated):
            operation()
