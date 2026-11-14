"""Real native source, proof, codec-ancestry and independent-RNG qualification."""

# ruff: noqa: D103
from random import Random
from typing import Any

import pytest
from atomheart.games.morpion import MorpionDynamics, initial_state
from valanga.evaluations import Certainty

from anemone import create_search_with_tree_eval_factory
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    load_search_from_checkpoint_payload,
)
from anemone.checkpoints.rng_state import rollout_random_generator
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.prepared_evaluator import (
    SourceIdentity,
    SupportedSearchRecipe,
    freeze_evaluator_source,
    prepare_evaluator_context,
    stage_evaluator_context,
)
from anemone.prepared_evaluator.native import decode, encode
from anemone.tree_manager.opening_expansion_config import (
    OpeningExpansionConfig,
    OpeningExpansionKind,
    RolloutActionSelectorKind,
    RolloutExpansionConfig,
)
from tests.morpion_refresh_helpers import Dynamics, State
from tests.test_prepared_evaluator import (
    IDENTITY,
    INDEX_RECIPES,
    Candidate,
    Codec,
    Incumbent,
    args,
    limits,
    native_bytes,
)


def configuration(rollout: bool, index_computation: Any = INDEX_RECIPES[1]) -> Any:
    result = args(index_computation)
    if rollout:
        result.opening_expansion = OpeningExpansionConfig(
            kind=OpeningExpansionKind.ROLLOUT,
            rollout=RolloutExpansionConfig(
                max_extra_steps=1,
                action_selector_kind=RolloutActionSelectorKind.RANDOM_OPENABLE,
                random_seed=19,
            ),
        )
    return result


def source_case(
    *,
    rollout: bool = False,
    steps: int = 5,
    state: State | None = None,
    index_computation: Any = INDEX_RECIPES[1],
) -> Any:
    config = configuration(rollout, index_computation)
    owner = create_search_with_tree_eval_factory(
        state_type=State,
        dynamics=Dynamics(),
        starting_state=state or State(initial_state()),
        args=config,
        random_generator=Random(7),
        master_state_evaluator=Incumbent(),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    for _ in range(steps):
        owner.step()
    recipe = SupportedSearchRecipe(
        opening_expansion=config.opening_expansion,
        index_computation=index_computation,
        rng_layout="independent_rollout" if rollout else "search_only",
    )
    return owner, config, recipe


def pinned(payload: Any, recipe: SupportedSearchRecipe) -> Any:
    return freeze_evaluator_source(
        payload,
        codec_type=Codec,
        identity=SourceIdentity("head", "incumbent", "codec", "build", "morpion"),
        recipe=recipe,
        limits=limits(),
    )


def restored(source: Any, config: Any) -> Any:
    return load_search_from_checkpoint_payload(
        decode(source._checkpoint_bytes),
        state_codec=Codec(),
        dynamics=Dynamics(),
        args=config,
        state_type=State,
        master_state_value_evaluator=Incumbent(),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )


def prepare_install(
    source: Any, config: Any, candidate: Any = None, validator: Any = None
) -> Any:
    candidate = (
        Candidate({i: float(i + 10) for i in range(512)})
        if candidate is None
        else candidate
    )
    prepared = prepare_evaluator_context(
        source,
        evaluator=candidate,
        evaluation_identity=IDENTITY,
        limits=limits(),
        validate_direct_certificate=validator,
    )
    owner = restored(source, config)
    plan = stage_evaluator_context(
        owner,
        current_source=source,
        prepared=prepared,
        evaluator=candidate,
        evaluation_identity=IDENTITY,
        limits=limits(),
    )
    return owner, plan, prepared, candidate


@pytest.mark.parametrize("rollout", [False, True])
@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_transpositions_independent_rngs_and_cold_continuation(
    rollout: bool, index_computation: Any
) -> None:
    live, config, recipe = source_case(
        index_computation=index_computation, rollout=rollout
    )
    assert any(n.parent_count() > 1 for n in live._iter_nodes_by_depth())
    source = pinned(build_search_checkpoint_payload(live, state_codec=Codec()), recipe)
    owner, plan, _, candidate = prepare_install(source, config)
    reference = restored(source, config)
    reference.refresh_with_evaluator(candidate, scope="all")
    search_rng = owner.node_selector.base.random_generator
    rollout_rng = rollout_random_generator(owner)
    before = (
        search_rng.getstate(),
        None if rollout_rng is None else rollout_rng.getstate(),
    )
    plan.install()
    assert owner.node_selector.base.random_generator is search_rng
    assert rollout_random_generator(owner) is rollout_rng
    assert before == (
        search_rng.getstate(),
        None if rollout_rng is None else rollout_rng.getstate(),
    )
    assert native_bytes(owner) == native_bytes(reference)
    for _ in range(3):
        count = owner.tree.branch_count
        owner.step()
        reference.step()
        assert owner.tree.branch_count > count
        assert native_bytes(owner) == native_bytes(reference)
        assert (
            owner.node_selector.base.random_generator.getstate()
            == reference.node_selector.base.random_generator.getstate()
        )
        if rollout_rng is not None:
            assert rollout_rng is not search_rng
            assert (
                rollout_rng.getstate() == rollout_random_generator(reference).getstate()
            )
    # Actual native cold restoration and a further nonzero step after installation.
    new_payload = build_search_checkpoint_payload(owner, state_codec=Codec())
    cold = load_search_from_checkpoint_payload(
        new_payload,
        state_codec=Codec(),
        dynamics=Dynamics(),
        args=config,
        state_type=State,
        master_state_value_evaluator=candidate,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    owner.step()
    cold.step()
    assert native_bytes(owner) == native_bytes(cold)


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_alternate_codec_parent_is_distinct_from_graph_parent(
    index_computation: Any,
) -> None:
    live, config, recipe = source_case(index_computation=index_computation, steps=6)
    payload = build_search_checkpoint_payload(live, state_codec=Codec())
    nodes = {n.id: n for n in live._iter_nodes_by_depth()}
    record = next(n for n in payload.tree.nodes if nodes[n.node_id].parent_count() > 1)
    node = nodes[record.node_id]
    parent, _ = next(
        (p, b)
        for p, b in node.iter_parent_items()
        if p.id != record.state_payload.state_parent_node_id
    )
    # Legal transpositions share board identity but can have different move
    # histories. Keep the codec's valid native delta ancestry; put the other
    # real incoming parent first in the graph's representative parent order.
    record.parent_node_id = parent.id
    parent_record = next(n for n in payload.tree.nodes if n.node_id == parent.id)
    record.branch_from_parent = next(
        link.branch_key
        for link in parent_record.linked_children
        if link.child_node_id == record.node_id
    )
    record.parent_node_order = [parent.id] + [
        p.id for p, _ in node.iter_parent_items() if p.id != parent.id
    ]
    assert record.parent_node_id != record.state_payload.state_parent_node_id
    source = pinned(payload, recipe)
    owner, plan, _, candidate = prepare_install(source, config)
    reference = restored(source, config)
    reference.refresh_with_evaluator(candidate, scope="all")
    plan.install()
    assert native_bytes(owner) == native_bytes(reference)
    owner.step()
    reference.step()
    assert native_bytes(owner) == native_bytes(reference)


def endgame() -> State:
    state = initial_state()
    dynamics = MorpionDynamics()
    for _ in range(200):
        actions = dynamics.all_legal_actions(state)
        result = dynamics.step(state, actions[0])
        if result.is_over:
            return State(state)
        state = result.next_state
    message = "real endgame did not terminate"
    raise AssertionError(message)


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_terminal_forced_and_estimate_parity(index_computation: Any) -> None:
    live, config, recipe = source_case(
        index_computation=index_computation, steps=1, state=endgame()
    )
    payload = build_search_checkpoint_payload(live, state_codec=Codec())
    source = pinned(payload, recipe)
    owner, plan, context, candidate = prepare_install(
        source, config, Candidate({0: 42})
    )
    reference = restored(source, config)
    reference.refresh_with_evaluator(candidate, scope="all")
    plan.install()
    assert native_bytes(owner) == native_bytes(reference)
    root = owner.tree.root_node.tree_evaluation
    assert (
        root.direct_value.score == 42
        and root.direct_value.certainty == Certainty.ESTIMATE
    )
    assert root.tree_value.score == 35 and root.tree_value.certainty == Certainty.FORCED
    assert context.skipped_terminal_count >= 1


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_direct_forced_requires_readonly_domain_evidence(
    index_computation: Any,
) -> None:
    live, _, recipe = source_case(
        index_computation=index_computation, steps=1, state=endgame()
    )
    live.tree.root_node.tree_evaluation.direct_value = (
        live.tree.root_node.tree_evaluation.tree_value
    )
    source = pinned(build_search_checkpoint_payload(live, state_codec=Codec()), recipe)
    with pytest.raises(ValueError, match="FORCED"):
        prepare_evaluator_context(
            source, evaluator=Candidate(), evaluation_identity=IDENTITY, limits=limits()
        )
    context = prepare_evaluator_context(
        source,
        evaluator=Candidate(),
        evaluation_identity=IDENTITY,
        validate_direct_certificate=lambda node: (
            node.id == 0 and node.tree_evaluation.direct_value.score == 35
        ),
        limits=limits(),
    )
    assert context.skipped_certified_count >= 1


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_current_head_requires_full_rebuild_but_raw_predictions_can_be_reused(
    index_computation: Any,
) -> None:
    class CachedCandidate(Candidate):
        """Cache exact scalar inputs under the complete frozen evaluation identity."""

        def __init__(self) -> None:
            super().__init__()
            self.cache = {}
            self.hits = 0
            self.misses = 0

        def evaluate_batch_items(self, items: Any) -> Any:
            result = []
            for node in items:
                # Full inline state input, not node id or depth: depth scaling
                # remains the responsibility of the shared native kernel.
                key = (
                    self.evaluation_identity,
                    encode(Codec().dump_anchor_ref(node.state)),
                )
                if key in self.cache:
                    self.hits += 1
                else:
                    self.cache[key] = super().evaluate_batch_items([node])[0]
                    self.misses += 1
                result.append(self.cache[key])
            return result

    live, config, recipe = source_case(index_computation=index_computation, steps=1)
    earlier = pinned(build_search_checkpoint_payload(live, state_codec=Codec()), recipe)
    candidate = CachedCandidate()
    old = prepare_evaluator_context(
        earlier, evaluator=candidate, evaluation_identity=IDENTITY, limits=limits()
    )
    live.step()
    live.step()
    current = pinned(build_search_checkpoint_payload(live, state_codec=Codec()), recipe)
    owner = restored(current, config)
    before = native_bytes(owner)
    with pytest.raises(ValueError, match="stale prepared"):
        stage_evaluator_context(
            owner,
            current_source=current,
            prepared=old,
            evaluator=candidate,
            evaluation_identity=IDENTITY,
            limits=limits(),
        )
    assert native_bytes(owner) == before
    misses = candidate.misses
    prepared = prepare_evaluator_context(
        current, evaluator=candidate, evaluation_identity=IDENTITY, limits=limits()
    )
    assert candidate.hits > 0 and candidate.misses > misses
    plan = stage_evaluator_context(
        owner,
        current_source=current,
        prepared=prepared,
        evaluator=candidate,
        evaluation_identity=IDENTITY,
        limits=limits(),
    )
    reference = restored(current, config)
    reference.refresh_with_evaluator(Candidate(), scope="all")
    plan.install()
    assert native_bytes(owner) == native_bytes(reference)


@pytest.mark.parametrize("index_computation", INDEX_RECIPES)
def test_aliased_rollout_rng_is_refused_before_mutation(index_computation: Any) -> None:
    live, config, recipe = source_case(
        index_computation=index_computation, rollout=True
    )
    source = pinned(build_search_checkpoint_payload(live, state_codec=Codec()), recipe)
    owner, _, prepared, candidate = prepare_install(source, config)
    owner.tree_manager.opening_expansion_executor.rollout_action_selector.random_generator = owner.node_selector.base.random_generator
    before = native_bytes(owner)
    with pytest.raises(ValueError, match="alias/ownership"):
        stage_evaluator_context(
            owner,
            current_source=source,
            prepared=prepared,
            evaluator=candidate,
            evaluation_identity=IDENTITY,
            limits=limits(),
        )
    assert native_bytes(owner) == before
