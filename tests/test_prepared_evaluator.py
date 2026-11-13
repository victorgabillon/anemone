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

    prepared_domain = "morpion-inline-v1"


class Candidate(Evaluator):
    """Bound deterministic native scalar evaluator identity."""

    evaluation_identity = IDENTITY


def limits() -> PreparationLimits:
    return PreparationLimits(monotonic() + 30)


def args() -> Any:
    result = _build_linoo_args()
    result.index_computation = IndexComputationType.MIN_GLOBAL_CHANGE
    result.stopping_criterion.tree_branch_limit = 4096
    result.node_selector.base.depth_selection_policy = "alternating_by_step"
    return result


def runtime(steps: int = 3) -> Any:
    search = create_search_with_tree_eval_factory(
        state_type=State,
        dynamics=Dynamics(),
        starting_state=State(initial_state()),
        args=args(),
        random_generator=Random(7),
        master_state_evaluator=Evaluator(),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    for _ in range(steps):
        search.step()
    return search


def cold(payload: Any) -> Any:
    return load_search_from_checkpoint_payload(
        payload,
        state_codec=Codec(),
        dynamics=Dynamics(),
        args=args(),
        state_type=State,
        master_state_value_evaluator=Evaluator(),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )


def freeze(payload: Any) -> Any:
    return freeze_evaluator_source(
        payload,
        codec_type=Codec,
        identity=SourceIdentity(
            "publication", "incumbent", "morpion-inline-v1", "test-build"
        ),
        recipe=SupportedSearchRecipe(),
        limits=limits(),
    )


def test_real_morpion_detached_refresh_matches_cold_live() -> None:
    payload = build_search_checkpoint_payload(runtime(), state_codec=Codec())
    source = freeze(payload)
    original = source._checkpoint_bytes
    candidate = Candidate({i: 42.0 + i for i in range(512)})
    context = prepare_evaluator_context(
        source, evaluator=candidate, evaluation_identity=IDENTITY, limits=limits()
    )
    reference = cold(payload)
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
def test_source_refuses_incomplete_or_unsupported_input(fault: str) -> None:
    payload = build_search_checkpoint_payload(runtime(), state_codec=Codec())
    if fault == "missing-parent":
        payload.tree.nodes[1].parent_node_id = 999999
    elif fault == "cycle":
        node = payload.tree.nodes[1]
        node.state_payload = replace(
            node.state_payload, state_parent_node_id=node.node_id
        )
    elif fault == "index":
        payload.tree.nodes[1].exploration_index = None
    elif fault == "selector":
        payload.selector_state = None
    with pytest.raises(ValueError):
        freeze_evaluator_source(
            payload,
            codec_type=BaseCodec if fault == "codec" else Codec,
            identity=SourceIdentity("p", "i", "c", "b"),
            recipe=SupportedSearchRecipe(),
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
