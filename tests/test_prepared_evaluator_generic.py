"""A small non-Morpion counting game exercises the generic owned-inline contract."""

# ruff: noqa: D102, D103
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any

import pytest
from valanga import SOLO, Outcome, OverEvent, Transition
from valanga.evaluations import Certainty, Value

from anemone import create_search_with_tree_eval_factory
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    load_sharded_search_checkpoint,
    write_sharded_search_checkpoint,
)
from anemone.node_evaluation.full_refresh import iter_nodes
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.prepared_evaluator import (
    OwnedInlineCodecCapabilities,
    SourceIdentity,
    SupportedSearchRecipe,
    freeze_evaluator_source,
    prepare_evaluator_context,
    stage_evaluator_context,
)
from anemone.prepared_evaluator.native import derived, encode, reconstruct
from tests.fake_yaml_game import FakeBranchKeyGenerator
from tests.test_checkpoint_materialize import read_limits
from tests.test_prepared_evaluator import IDENTITY, INDEX_RECIPES, args, limits


@dataclass
class CounterState:
    """Choose zero or one for eight turns; distinct paths can meet at one state."""

    depth: int = 0
    total: int = 0
    scratch: list[int] = field(default_factory=list)
    turn: Any = SOLO

    @property
    def tag(self) -> int:
        return self.depth * 100 + self.total

    def is_game_over(self) -> bool:
        return self.depth == 8

    def pprint(self) -> str:
        return str(self.tag)


class CounterDynamics:
    """A deterministic layered single-agent counting game, without board APIs."""

    def legal_actions(self, state: CounterState) -> Any:
        return FakeBranchKeyGenerator([] if state.is_game_over() else [0, 1])

    def step(self, state: CounterState, action: int) -> Any:
        child = CounterState(state.depth + 1, state.total + action)
        return Transition(
            next_state=child,
            modifications=None,
            is_over=child.is_game_over(),
            over_event=None,
            info={},
        )

    def action_name(self, state: CounterState, action: int) -> str:
        return str(action)

    def action_from_name(self, state: CounterState, name: str) -> int:
        return int(name)


class CounterCodec:
    """Fresh inline records contain just the counter; no external game objects."""

    prepared_capabilities = OwnedInlineCodecCapabilities("counter-state-v1")

    def __init__(self) -> None:
        """Own a fresh decode-observation list per instance."""
        self.decoded: list[CounterState] = []

    def dump_anchor_ref(self, state: CounterState) -> Any:
        return [state.depth, state.total]

    def dump_delta_from_parent(
        self,
        *,
        parent_state: CounterState,
        child_state: CounterState,
        branch_from_parent: Any = None,
    ) -> Any:
        return child_state.total - parent_state.total

    def load_anchor_ref(self, ref: Any) -> CounterState:
        state = CounterState(*ref)
        self.decoded.append(state)
        return state

    def load_child_from_delta(
        self,
        *,
        parent_state: CounterState,
        delta_ref: Any,
        branch_from_parent: Any = None,
    ) -> CounterState:
        assert delta_ref == branch_from_parent
        state = CounterState(parent_state.depth + 1, parent_state.total + delta_ref)
        self.decoded.append(state)
        return state

    def dump_state_summary(self, state: CounterState) -> Any:
        return {"tag": state.tag, "is_terminal": state.is_game_over()}


class CounterEvaluator:
    """Strict scalar counter value, with native terminal provenance."""

    def __init__(self, candidate: bool = False) -> None:
        """Bind the strict deterministic candidate or incumbent."""
        self.evaluation_identity = IDENTITY if candidate else "incumbent"
        self.offset = 0.25 if candidate else 0.0
        self.over = self

    def check_obvious_over_events(self, state: CounterState) -> Any:
        if state.is_game_over():
            return OverEvent(
                outcome=Outcome.WIN, winner=None, termination="eight"
            ), float(state.total)
        return None, None

    def evaluate_batch_items(self, items: Any) -> list[Value]:
        return [
            Value(
                score=float(n.state.total) + self.offset, certainty=Certainty.ESTIMATE
            )
            for n in items
        ]


def fixture(index: Any, depth: int = 0) -> Any:
    live = create_search_with_tree_eval_factory(
        state_type=CounterState,
        dynamics=CounterDynamics(),
        starting_state=CounterState(depth=depth),
        args=args(index),
        random_generator=Random(7),
        master_state_evaluator=CounterEvaluator(),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    for _ in range(4):
        if live.tree.root_node.tree_evaluation.has_exact_value():
            break
        live.step()
    payload = build_search_checkpoint_payload(live, state_codec=CounterCodec())
    return live, payload


def freeze(payload: Any, index: Any, codec: Any = CounterCodec) -> Any:
    return freeze_evaluator_source(
        payload,
        codec_type=codec,
        identity=SourceIdentity(
            "counter-publication",
            "incumbent",
            "codec-build-1",
            "anemone-build",
            "counter",
        ),
        recipe=SupportedSearchRecipe(index_computation=index),
        limits=limits(),
    )


def checkpoint(search: Any) -> bytes:
    return encode(build_search_checkpoint_payload(search, state_codec=CounterCodec()))


@pytest.mark.parametrize("index", INDEX_RECIPES)
@pytest.mark.parametrize("layout", ["split", "node_records"])
def test_non_morpion_original_input_parity_and_ownership(
    tmp_path: Any, index: Any, layout: str
) -> None:
    live, payload = fixture(index)
    write_sharded_search_checkpoint(
        payload, tmp_path, layout=layout, node_count_per_shard=3
    )
    source = freeze(
        load_sharded_search_checkpoint(tmp_path, limits=read_limits()), index
    )
    candidate = CounterEvaluator(True)
    context = prepare_evaluator_context(
        source, evaluator=candidate, evaluation_identity=IDENTITY, limits=limits()
    )

    def cold() -> Any:
        return load_search_from_sharded_checkpoint(
            tmp_path,
            state_codec=CounterCodec(),
            dynamics=CounterDynamics(),
            args=args(index),
            state_type=CounterState,
            master_state_value_evaluator=CounterEvaluator(),
            node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
        )

    owner, reference = cold(), cold()
    arena = reconstruct(
        source._checkpoint_bytes,
        CounterCodec,
        source.recipe,
        limits(),
        capability=source.codec_capability,
    )
    live_states = [n.state for n in live._iter_nodes_by_depth()]
    owner_states = [n.state for n in owner._iter_nodes_by_depth()]
    detached = [n.state for n in iter_nodes(arena.tree)]
    for state in detached:
        assert all(
            state is not other and state.scratch is not other.scratch
            for other in live_states + owner_states
        )
    detached[0].scratch.append(99)
    assert all(not s.scratch for s in live_states + owner_states)
    first = next(iter(iter_nodes(arena.tree)))
    assert (
        first.state_handle.resolver.state_codec.codec
        is not owner.tree.root_node.state_handle.resolver.state_codec
    )
    reference.refresh_with_evaluator(candidate, scope="all")
    assert context._derived_bytes == encode(
        derived(reference.tree, reference.node_selector.base)
    )
    stage_evaluator_context(
        owner,
        current_source=source,
        prepared=context,
        evaluator=candidate,
        evaluation_identity=IDENTITY,
        limits=limits(),
    ).install()
    assert checkpoint(owner) == checkpoint(reference)
    for _ in range(2):
        before = owner.tree.branch_count
        owner.step()
        reference.step()
        assert owner.tree.branch_count > before
        assert checkpoint(owner) == checkpoint(reference)


@pytest.mark.parametrize(
    "fault",
    ["absent", "empty", "external", "mapping", "factory", "instance", "singleton"],
)
def test_generic_capability_refusals(fault: str) -> None:
    _, payload = fixture(None)

    class Bad(CounterCodec):
        pass

    if fault == "absent":
        Bad.prepared_capabilities = None
    elif fault == "empty":
        Bad.prepared_capabilities = OwnedInlineCodecCapabilities(" ")
    elif fault == "external":
        Bad.prepared_capabilities = OwnedInlineCodecCapabilities(
            "counter", "external-store"
        )
    elif fault == "mapping":
        Bad.prepared_capabilities = {
            "schema_identity": "counter",
            "reconstruction": "owned-inline-v1",
        }
    elif fault == "singleton":
        Bad.__new__ = staticmethod(lambda cls: CounterCodec())
    codec = (
        (lambda: CounterCodec())
        if fault == "factory"
        else CounterCodec()
        if fault == "instance"
        else Bad
    )
    with pytest.raises(ValueError, match=r"codec|capability"):
        freeze(payload, None, codec)


@pytest.mark.parametrize(
    "fault",
    [
        "summary",
        "terminal",
        "parent-alias",
        "capability-drift",
        "source",
        "owner",
        "owner-alias",
    ],
)
@pytest.mark.parametrize("index", INDEX_RECIPES)
def test_generic_source_and_owner_refusals(
    tmp_path: Any, monkeypatch: Any, fault: str, index: Any
) -> None:
    _, payload = fixture(index)
    if fault in {"summary", "terminal"}:
        summary = payload.tree.nodes[0].state_payload.state_summary
        summary["tag" if fault == "summary" else "is_terminal"] = (
            -1 if fault == "summary" else True
        )
        with pytest.raises(ValueError, match="summary"):
            freeze(payload, index)
        return
    if fault == "parent-alias":

        class Aliasing(CounterCodec):
            def load_child_from_delta(self, *, parent_state: Any, **kwargs: Any) -> Any:
                return parent_state

        with pytest.raises(ValueError, match="alias"):
            freeze(payload, index, Aliasing)
        return
    source = freeze(payload, index)
    candidate = CounterEvaluator(True)
    if fault == "capability-drift":
        monkeypatch.setattr(
            CounterCodec,
            "prepared_capabilities",
            OwnedInlineCodecCapabilities("counter-v2"),
        )
        with pytest.raises(ValueError, match="capability changed"):
            prepare_evaluator_context(
                source,
                evaluator=candidate,
                evaluation_identity=IDENTITY,
                limits=limits(),
            )
        return
    context = prepare_evaluator_context(
        source, evaluator=candidate, evaluation_identity=IDENTITY, limits=limits()
    )
    write_sharded_search_checkpoint(payload, tmp_path, layout="split")
    owner = load_search_from_sharded_checkpoint(
        tmp_path,
        state_codec=CounterCodec(),
        dynamics=CounterDynamics(),
        args=args(index),
        state_type=CounterState,
        master_state_value_evaluator=CounterEvaluator(),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    if fault == "source":
        source = replace(
            source, identity=replace(source.identity, domain="different-namespace")
        )
    elif fault == "owner":
        owner.tree.root_node.state_handle.resolver.state_codec.prepared_capabilities = (
            OwnedInlineCodecCapabilities("other-schema")
        )
    else:
        cached = {n.state.tag: n.state for n in owner._iter_nodes_by_depth()}
        monkeypatch.setattr(
            CounterCodec,
            "load_anchor_ref",
            lambda self, ref: cached[ref[0] * 100 + ref[1]],
        )
        monkeypatch.setattr(
            CounterCodec,
            "load_child_from_delta",
            lambda self, *, parent_state, delta_ref, branch_from_parent=None: cached[
                (parent_state.depth + 1) * 100 + parent_state.total + delta_ref
            ],
        )
    before = checkpoint(owner)
    with pytest.raises(
        ValueError, match=r"source identity|codec/handle|aliases destination"
    ):
        stage_evaluator_context(
            owner,
            current_source=source,
            prepared=context,
            evaluator=candidate,
            evaluation_identity=IDENTITY,
            limits=limits(),
        )
    assert checkpoint(owner) == before


@pytest.mark.parametrize("index", INDEX_RECIPES)
def test_non_morpion_terminal_and_partial_updates(index: Any) -> None:
    from anemone.value_updates import NodeValueUpdate

    live, payload = fixture(index, depth=6)
    source = freeze(payload, index)
    context = prepare_evaluator_context(
        source,
        evaluator=CounterEvaluator(True),
        evaluation_identity=IDENTITY,
        limits=limits(),
    )
    assert context.skipped_terminal_count > 0
    # The ordinary selected/scoped paths remain available without a prepared context.
    live, _ = fixture(index)
    live.set_evaluator(CounterEvaluator(True))
    report = live.reevaluate_nodes([live.tree.root_node])
    assert report.requested_count == 1 and report.reevaluated_count == 1
    update = live.apply_node_value_updates(
        [NodeValueUpdate(node_id=str(live.tree.root_node.id), direct_value=0.375)]
    )
    assert update is not None
    live.refresh_with_evaluator(CounterEvaluator(), scope="root_children")


def test_budget_codec_accepts_protocol_keyword_and_preserves_decode_limit() -> None:
    from anemone.prepared_evaluator.native import _BudgetCodec

    codec = CounterCodec()
    bounded = _BudgetCodec(codec, replace(limits(), max_state_decodes=1))
    state = bounded.load_anchor_ref(anchor_ref=[2, 1])
    assert state.tag == 201
    assert bounded.count == 1
    assert codec.decoded == [state]
    with pytest.raises(ValueError, match="state decode bound exceeded"):
        bounded.load_anchor_ref(anchor_ref=[2, 1])
    assert codec.decoded == [state]
