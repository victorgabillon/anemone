"""Finite index recipes preserve native absence, history and refusal boundaries."""

# ruff: noqa: D103
import json
from dataclasses import replace
from typing import Any

import pytest

from anemone.checkpoints import build_search_checkpoint_payload
from anemone.checkpoints.payloads import ExplorationIndexCheckpointPayload
from anemone.indices.index_manager.factory import create_exploration_index_manager
from anemone.indices.node_indices.index_types import IndexComputationType
from anemone.prepared_evaluator import (
    PreparedEvaluatorRefusal,
    prepare_evaluator_context,
)
from anemone.prepared_evaluator.native import decode, derived, encode, reconstruct
from tests.test_prepared_evaluator import (
    IDENTITY,
    INDEX_RECIPES,
    Codec,
    freeze,
    limits,
    native_bytes,
    prepared_case,
    runtime,
    stage,
)


def test_null_path_never_constructs_minmax_placeholders(monkeypatch: Any) -> None:
    from anemone.indices.node_indices.index_data import MinMaxPathValue

    def forbidden(*args: Any, **kwargs: Any) -> None:
        message = "null recipe constructed a MinMaxPathValue"
        raise AssertionError(message)

    monkeypatch.setattr(MinMaxPathValue, "__init__", forbidden)
    source, context, candidate, owner = prepared_case(None)
    stage(source, context, candidate, owner).install()
    branches = owner.tree.branch_count
    owner.step()
    assert owner.tree.branch_count > branches
    assert all(n.exploration_index_data is None for n in owner._iter_nodes_by_depth())


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_recipe_and_complete_selector_history_survive_reconstruction(
    index_computation: Any,
) -> None:
    source, context, candidate, owner = prepared_case(index_computation)
    payload = decode(source._checkpoint_bytes)
    arena = reconstruct(source._checkpoint_bytes, Codec, source.recipe, limits())
    assert source.recipe.index_computation is index_computation
    assert payload.selector_state.selection_step_count == 3
    assert encode(derived(arena.tree, arena.selector)["selector"]) == encode(
        payload.selector_state
    )
    data = json.loads(context._derived_bytes)
    assert data["selector"]["selection_step_count"] == 3
    assert all((n[2] is None) == (index_computation is None) for n in data["nodes"])
    stage(source, context, candidate, owner).install()
    installed = build_search_checkpoint_payload(owner, state_codec=Codec())
    assert installed.selector_state.selection_step_count == 3
    assert all(
        (n.exploration_index is None) == (index_computation is None)
        for n in installed.tree.nodes
    )
    if index_computation is None:
        assert all(
            n.exploration_index_data is None for n in owner._iter_nodes_by_depth()
        )


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
@pytest.mark.parametrize("fault", ["opposite-recipe", "mixed", "fake-null"])
def test_source_rejects_wrong_or_mixed_index_coverage(
    index_computation: Any,
    fault: str,
) -> None:
    payload = build_search_checkpoint_payload(
        runtime(index_computation=index_computation), state_codec=Codec()
    )
    if fault == "opposite-recipe":
        index_computation = INDEX_RECIPES[index_computation is None]
    elif fault == "fake-null":
        payload.tree.nodes[1].exploration_index = ExplorationIndexCheckpointPayload()
    else:
        payload.tree.nodes[1].exploration_index = (
            ExplorationIndexCheckpointPayload(kind="MinMaxPathValue", payload={})
            if index_computation is None
            else None
        )
    with pytest.raises(PreparedEvaluatorRefusal, match="index payload/recipe"):
        freeze(payload, index_computation)


@pytest.mark.parametrize(
    "unsupported",
    [
        IndexComputationType.MIN_LOCAL_CHANGE,
        IndexComputationType.RECUR_ZIPF,
        "min_global_change",
        False,
    ],
)
def test_only_exact_finite_index_recipes_are_admitted(unsupported: Any) -> None:
    payload = build_search_checkpoint_payload(runtime(), state_codec=Codec())
    with pytest.raises(PreparedEvaluatorRefusal, match="unsupported index recipe"):
        freeze(payload, unsupported)


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
@pytest.mark.parametrize("artifact", ["source", "context", "candidate-source"])
def test_index_artifact_mismatch_refuses_before_inference_or_mutation(
    index_computation: Any,
    artifact: str,
    monkeypatch: Any,
) -> None:
    source, context, candidate, owner = prepared_case(index_computation)
    before = native_bytes(owner)
    opposite = replace(
        source,
        recipe=replace(
            source.recipe, index_computation=INDEX_RECIPES[index_computation is None]
        ),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        message = "inference before index refusal"
        raise AssertionError(message)

    monkeypatch.setattr(candidate, "evaluate_batch_items", forbidden)
    with pytest.raises(PreparedEvaluatorRefusal):
        if artifact == "source":
            prepare_evaluator_context(
                opposite,
                evaluator=candidate,
                evaluation_identity=IDENTITY,
                limits=limits(),
            )
        elif artifact == "candidate-source":
            stage(source, replace(context, source=opposite), candidate, owner)
        else:
            data = json.loads(context._derived_bytes)
            data["nodes"][1][2] = (
                {"kind": "MinMaxPathValue", "payload": {}}
                if index_computation is None
                else None
            )
            stage(
                source, replace(context, _derived_bytes=encode(data)), candidate, owner
            )
    assert native_bytes(owner) == before
    assert not owner._prepared_evaluator_invalidated


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
@pytest.mark.parametrize("when", ["before-stage", "after-stage"])
@pytest.mark.parametrize("fault", ["manager", "factory", "node"])
def test_owner_index_recipe_drift_refuses_before_installation(
    index_computation: Any,
    when: str,
    fault: str,
) -> None:
    source, context, candidate, owner = prepared_case(index_computation)
    plan = stage(source, context, candidate, owner) if when == "after-stage" else None
    opposite = INDEX_RECIPES[index_computation is None]
    if fault == "manager":
        owner.tree_manager.index_manager = create_exploration_index_manager(opposite)
    elif fault == "factory":
        owner.tree_manager.tree_manager.node_factory.exploration_index_data_create.__self__.index_computation = opposite
    else:
        from anemone.indices.node_indices.factory import create_exploration_index_data

        node = owner.tree.root_node
        node.exploration_index_data = create_exploration_index_data(
            node.tree_node, opposite
        )
    before = native_bytes(owner)
    with pytest.raises(PreparedEvaluatorRefusal):
        if plan is None:
            stage(source, context, candidate, owner)
        else:
            plan.install()
    assert native_bytes(owner) == before
    assert not owner._prepared_evaluator_invalidated
