"""Independent review regressions for prepared artifact admission."""

# ruff: noqa: D103
from dataclasses import replace
from typing import Any

import pytest

from anemone.prepared_evaluator import (
    PreparedEvaluatorRefusal,
    prepare_evaluator_context,
    stage_evaluator_context,
)
from anemone.prepared_evaluator.native import decode, encode
from tests.test_prepared_evaluator import (
    IDENTITY,
    INDEX_RECIPES,
    freeze,
    limits,
    native_bytes,
    prepared_case,
)


@pytest.mark.parametrize("operation", ["prepare", "stage"])
@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_smaller_artifact_budget_refuses_before_inference_or_staging(
    operation: str, monkeypatch: Any, index_computation: Any
) -> None:
    source, context, candidate, owner = prepared_case(index_computation)
    before = native_bytes(owner)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        message = "inference ran before artifact-size refusal"
        raise AssertionError(message)

    monkeypatch.setattr(candidate, "evaluate_batch_items", forbidden)
    with pytest.raises(PreparedEvaluatorRefusal, match="payload bound"):
        if operation == "prepare":
            prepare_evaluator_context(
                source,
                evaluator=candidate,
                evaluation_identity=IDENTITY,
                limits=replace(limits(), max_payload_bytes=1),
            )
        else:
            stage_evaluator_context(
                owner,
                current_source=source,
                prepared=context,
                evaluator=candidate,
                evaluation_identity=IDENTITY,
                limits=replace(limits(), max_payload_bytes=1),
            )
    assert native_bytes(owner) == before


@pytest.mark.parametrize("artifact", ["source", "context"])
@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_saved_atoms_cannot_request_arbitrary_module_import(
    artifact: str, monkeypatch: Any, index_computation: Any
) -> None:
    from anemone.checkpoints import value_serialization

    source, context, candidate, owner = prepared_case(index_computation)
    malicious = {
        "type": "enum",
        "module": "unapproved_payload_module",
        "qualname": "Surprise",
        "name": "VALUE",
    }

    def forbidden(name: str) -> Any:
        raise AssertionError("payload requested import: " + name)

    monkeypatch.setattr(value_serialization, "import_module", forbidden)
    before = native_bytes(owner)
    with pytest.raises(PreparedEvaluatorRefusal, match="enum"):
        if artifact == "source":
            payload = decode(source._checkpoint_bytes)
            payload.tree.nodes[0].linked_children[0] = replace(
                payload.tree.nodes[0].linked_children[0], branch_key=malicious
            )
            freeze(payload, index_computation)
        else:
            import json

            derived = json.loads(context._derived_bytes)
            derived["nodes"][0][1]["direct_value"]["line"] = [malicious]
            context = replace(context, _derived_bytes=encode(derived))
            stage_evaluator_context(
                owner,
                current_source=source,
                prepared=context,
                evaluator=candidate,
                evaluation_identity=IDENTITY,
                limits=limits(),
            )
    assert native_bytes(owner) == before


@pytest.mark.parametrize(
    "policy", ["inverse_depth", "opened_count_depth_index", "alternating_by_step"]
)
@pytest.mark.parametrize(
    "rollout_kind", ["random_openable", "random_legal_prefer_openable"]
)
@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_admitted_selector_and_rollout_variants_match_native_cold(
    policy: str, rollout_kind: str, index_computation: Any
) -> None:
    from random import Random

    from atomheart.games.morpion import initial_state

    from anemone import create_search_with_tree_eval_factory
    from anemone.checkpoints import build_search_checkpoint_payload
    from anemone.checkpoints.rng_state import rollout_random_generator
    from anemone.node_evaluation.tree.single_agent.factory import (
        NodeMaxEvaluationFactory,
    )
    from anemone.prepared_evaluator import SupportedSearchRecipe
    from anemone.tree_manager.opening_expansion_config import RolloutActionSelectorKind
    from tests.morpion_refresh_helpers import Dynamics, State
    from tests.test_prepared_evaluator import Codec, Incumbent
    from tests.test_prepared_evaluator_continuation import (
        configuration,
        pinned,
        prepare_install,
        restored,
    )

    config = configuration(True, index_computation)
    config.node_selector.base.depth_selection_policy = policy
    config.opening_expansion = replace(
        config.opening_expansion,
        rollout=replace(
            config.opening_expansion.rollout,
            action_selector_kind=RolloutActionSelectorKind(rollout_kind),
        ),
    )
    live = create_search_with_tree_eval_factory(
        state_type=State,
        dynamics=Dynamics(),
        starting_state=State(initial_state()),
        args=config,
        random_generator=Random(7),
        master_state_evaluator=Incumbent(),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    for _ in range(3):
        live.step()
    recipe = SupportedSearchRecipe(
        depth_selection_policy=policy,
        index_computation=index_computation,
        opening_expansion=config.opening_expansion,
        rng_layout="independent_rollout",
    )
    source = pinned(build_search_checkpoint_payload(live, state_codec=Codec()), recipe)
    owner, plan, _, candidate = prepare_install(source, config)
    reference = restored(source, config)
    reference.refresh_with_evaluator(candidate, scope="all")
    rng = rollout_random_generator(owner)
    search_rng = owner.node_selector.base.random_generator
    before = (rng.getstate(), search_rng.getstate())
    plan.install()
    assert rng is rollout_random_generator(owner) and rng is not search_rng
    assert before == (rng.getstate(), search_rng.getstate())
    assert native_bytes(owner) == native_bytes(reference)
    for _ in range(2):
        branches = owner.tree.branch_count
        owner.step()
        reference.step()
        assert owner.tree.branch_count > branches
        assert native_bytes(owner) == native_bytes(reference)


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_plan_disposal_and_dead_owner_refuse_without_retaining_runtime(
    index_computation: Any,
) -> None:
    import gc
    import weakref

    from tests.test_prepared_evaluator import stage

    source, prepared, candidate, owner = prepared_case(index_computation)
    plan = stage(source, prepared, candidate, owner)
    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert owner_ref() is None
    with pytest.raises(PreparedEvaluatorRefusal, match="no longer exists"):
        plan.install()
    plan.dispose()
    plan.dispose()
    with pytest.raises(PreparedEvaluatorRefusal, match="consumed"):
        plan.install()


@pytest.mark.parametrize("when", ["before-stage", "after-stage"])
@pytest.mark.parametrize("fault", ["callable", "index-kind", "depth-index"])
@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_future_node_index_factory_is_part_of_the_supported_recipe(
    when: str, fault: str, index_computation: Any
) -> None:
    from anemone.indices.node_indices.index_types import IndexComputationType
    from tests.test_prepared_evaluator import stage

    source, prepared, candidate, owner = prepared_case(index_computation)
    plan = stage(source, prepared, candidate, owner) if when == "after-stage" else None
    factory = owner.tree_manager.tree_manager.node_factory
    if fault == "callable":
        factory.exploration_index_data_create = lambda node: None
    elif fault == "index-kind":
        factory.exploration_index_data_create.__self__.index_computation = (
            IndexComputationType.MIN_LOCAL_CHANGE
        )
    else:
        factory.exploration_index_data_create.__self__.depth_index = True
    before = native_bytes(owner)
    with pytest.raises(PreparedEvaluatorRefusal):
        if plan is None:
            stage(source, prepared, candidate, owner)
        else:
            plan.install()
    assert native_bytes(owner) == before and not owner._prepared_evaluator_invalidated
