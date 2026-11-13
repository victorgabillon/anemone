"""Bounded refusals, repeated candidates and canonical writer qualification."""

# ruff: noqa: D103
from dataclasses import replace
from time import monotonic
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from anemone.checkpoints import (
    build_search_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    write_sharded_search_checkpoint_from_search,
)
from anemone.prepared_evaluator import (
    RuntimeInvalidated,
    freeze_evaluator_source,
    installation,
    prepare_evaluator_context,
    stage_evaluator_context,
)
from anemone.prepared_evaluator.native import decode
from anemone.tree_manager.opening_expansion_config import (
    OpeningExpansionConfig,
    OpeningExpansionKind,
)
from tests.test_prepared_evaluator import (
    IDENTITY,
    Candidate,
    Codec,
    cold,
    limits,
    native_bytes,
    prepared_case,
    stage,
)


@pytest.mark.parametrize(
    "fault", ["deadline", "cancel", "decodes", "payload", "recipe", "rollout-state"]
)
def test_bounded_freeze_refusal(fault: str) -> None:
    source, _, _, _ = prepared_case()
    payload = decode(source._checkpoint_bytes)
    budget = limits()
    recipe = source.recipe
    if fault == "deadline":
        budget = replace(budget, deadline_monotonic=monotonic() - 1)
    if fault == "cancel":
        budget = replace(budget, cancelled=lambda: True)
    if fault == "decodes":
        budget = replace(budget, max_state_decodes=1)
    if fault == "payload":
        budget = replace(budget, max_payload_bytes=1)
    if fault == "recipe":
        recipe = replace(
            recipe,
            opening_expansion=OpeningExpansionConfig(kind=OpeningExpansionKind.ROLLOUT),
        )
    if fault == "rollout-state":
        payload.rollout_rng_state = [3, [0], None]
    with pytest.raises(ValueError):
        freeze_evaluator_source(
            payload,
            codec_type=Codec,
            identity=source.identity,
            recipe=recipe,
            limits=budget,
        )


@pytest.mark.parametrize("fault", ["recipe", "model", "coverage", "cancel", "bindings"])
def test_stage_failure_leaves_complete_owner_unchanged(fault: str) -> None:
    source, prepared, candidate, owner = prepared_case()
    budget = limits()
    if fault == "recipe":
        owner.stopping_criterion.tree_branch_limit -= 1
    if fault == "model":
        candidate.evaluation_identity = replace(IDENTITY, weights="wrong")
    if fault == "coverage":
        prepared = replace(prepared, _derived_bytes=b'{"nodes":[],"selector":{}}')
    if fault == "cancel":
        budget = replace(budget, cancelled=lambda: True)
    plan = None
    if fault == "bindings":
        plan = stage(source, prepared, candidate, owner)
        owner._latest_step_report = object()
    before = native_bytes(owner)
    with pytest.raises(ValueError):
        if plan is not None:
            plan.install()
        else:
            stage_evaluator_context(
                owner,
                current_source=source,
                prepared=prepared,
                evaluator=candidate,
                evaluation_identity=IDENTITY,
                limits=budget,
            )
    assert native_bytes(owner) == before and not owner._prepared_evaluator_invalidated
    if plan is not None:
        assert not plan._nodes and plan._evaluator is None


def test_inference_failure_never_touches_owner() -> None:
    class Failing(Candidate):
        def evaluate_batch_items(self, items: Any) -> Any:
            message = "injected inference failure"
            raise RuntimeError(message)

    source, _, _, owner = prepared_case()
    before = native_bytes(owner)
    with pytest.raises(RuntimeError, match="injected inference"):
        prepare_evaluator_context(
            source, evaluator=Failing(), evaluation_identity=IDENTITY, limits=limits()
        )
    assert native_bytes(owner) == before


def test_repeated_candidate_preserves_structured_incumbent_identity() -> None:
    source, prepared, candidate, owner = prepared_case()
    stage(source, prepared, candidate, owner).install()
    # A newly committed checkpoint can bind the previous candidate's full identity.
    next_source = freeze_evaluator_source(
        build_search_checkpoint_payload(owner, state_codec=Codec()),
        codec_type=Codec,
        identity=replace(source.identity, publication="next", incumbent=IDENTITY),
        recipe=source.recipe,
        limits=limits(),
    )
    next_candidate = Candidate({i: float(i + 80) for i in range(512)})
    next_identity = replace(IDENTITY, weights="next-weights")
    next_candidate.evaluation_identity = next_identity
    next_context = prepare_evaluator_context(
        next_source,
        evaluator=next_candidate,
        evaluation_identity=next_identity,
        limits=limits(),
    )
    owner = cold(decode(next_source._checkpoint_bytes))
    owner.tree_manager.node_evaluator.master_state_value_evaluator = candidate
    reference = cold(decode(next_source._checkpoint_bytes))
    reference.refresh_with_evaluator(next_candidate, scope="all")
    stage_evaluator_context(
        owner,
        current_source=next_source,
        prepared=next_context,
        evaluator=next_candidate,
        evaluation_identity=next_identity,
        limits=limits(),
    ).install()
    assert native_bytes(owner) == native_bytes(reference)
    assert owner.evaluator_version == 2


@pytest.mark.parametrize("layout", ["node_records", "split"])
def test_native_streaming_export_exact_after_install(
    tmp_path: Path, layout: str
) -> None:
    source, prepared, candidate, owner = prepared_case()
    reference = cold(decode(source._checkpoint_bytes))
    reference.refresh_with_evaluator(candidate, scope="all")
    stage(source, prepared, candidate, owner).install()
    for name, search in [("installed", owner), ("cold", reference)]:
        write_sharded_search_checkpoint_from_search(
            search,
            tmp_path / name,
            state_codec=Codec(),
            encoding="jsonl",
            layout=layout,
            node_count_per_shard=3,
        )
    expected = {
        p.relative_to(tmp_path / "cold"): p.read_bytes()
        for p in (tmp_path / "cold").rglob("*")
        if p.is_file()
    }
    actual = {
        p.relative_to(tmp_path / "installed"): p.read_bytes()
        for p in (tmp_path / "installed").rglob("*")
        if p.is_file()
    }
    assert actual == expected
    from anemone.node_evaluation.tree.single_agent.factory import (
        NodeMaxEvaluationFactory,
    )
    from tests.morpion_refresh_helpers import Dynamics, State
    from tests.test_prepared_evaluator import args

    restored = load_search_from_sharded_checkpoint(
        tmp_path / "installed",
        state_codec=Codec(),
        dynamics=Dynamics(),
        args=args(),
        state_type=State,
        master_state_value_evaluator=candidate,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    owner.step()
    restored.step()
    assert native_bytes(owner) == native_bytes(restored)


def test_invalidated_runtime_refuses_streaming_before_creating_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source, prepared, candidate, owner = prepared_case()
    plan = stage(source, prepared, candidate, owner)
    original = installation._assign_node

    def fail(node: Any, staged: Any) -> None:
        original(node, staged)
        message = "injected after assignment"
        raise RuntimeError(message)

    monkeypatch.setattr(installation, "_assign_node", fail)
    with pytest.raises(RuntimeInvalidated):
        plan.install()
    output = tmp_path / "forbidden"
    with pytest.raises(RuntimeInvalidated):
        write_sharded_search_checkpoint_from_search(owner, output, state_codec=Codec())
    assert not output.exists()


def test_cancellation_after_inference_returns_no_context() -> None:
    cancelled = False

    class CancellingCandidate(Candidate):
        def evaluate_batch_items(self, items: Any) -> Any:
            nonlocal cancelled
            values = super().evaluate_batch_items(items)
            cancelled = True
            return values

    source, _, _, owner = prepared_case()
    before = native_bytes(owner)
    with pytest.raises(ValueError, match="cancelled"):
        prepare_evaluator_context(
            source,
            evaluator=CancellingCandidate(),
            evaluation_identity=IDENTITY,
            limits=replace(limits(), cancelled=lambda: cancelled),
        )
    assert native_bytes(owner) == before


def test_stochastic_candidate_is_refused_without_resetting_growth_rngs() -> None:
    import random

    class StochasticCandidate(Candidate):
        def evaluate_batch_items(self, items: Any) -> Any:
            random.random()
            return super().evaluate_batch_items(items)

    source, _, _, owner = prepared_case()
    before = native_bytes(owner)
    ambient = random.getstate()
    try:
        with pytest.raises(ValueError, match="consumed RNG"):
            prepare_evaluator_context(
                source,
                evaluator=StochasticCandidate(),
                evaluation_identity=IDENTITY,
                limits=limits(),
            )
        assert native_bytes(owner) == before
    finally:
        random.setstate(ambient)


@pytest.mark.parametrize(
    "fault", ["cached-input", "representation-factory", "factory-swap"]
)
def test_unsupported_or_changed_input_bindings_refuse(fault: str) -> None:
    from copy import copy

    source, prepared, candidate, owner = prepared_case()
    factory = owner.tree_manager.tree_manager.node_factory
    plan = None
    if fault == "factory-swap":
        plan = stage(source, prepared, candidate, owner)
        owner.tree_manager.tree_manager.node_factory = copy(factory)
    elif fault == "cached-input":
        owner.tree.root_node._state_representation = object()
    else:
        factory.state_representation_factory = object()
    before = native_bytes(owner)
    with pytest.raises(ValueError):
        if plan is None:
            stage(source, prepared, candidate, owner)
        else:
            plan.install()
    assert native_bytes(owner) == before
    assert not owner._prepared_evaluator_invalidated
