"""Complete refresh contracts checked against an independent single-agent oracle."""
# ruff: noqa: D103

from typing import Any

import pytest
from valanga.evaluations import Certainty, Value

from anemone import NodeValueUpdate
from anemone.backup_policies import ExplicitMaxBackupPolicy
from anemone.node_evaluation.direct.node_direct_evaluator import EvaluationQueries
from tests.scientific_refresh_helpers import (
    Evaluator,
    certifies,
    nodes,
    reference,
    runtime,
    snapshot,
)

_CHILDREN = {0: [1, 2], 1: [3], 2: [3], 3: [4, 5], 4: [6], 5: [], 6: []}
_VALUES = {i: float(i) for i in range(7)}


def topology(search: Any) -> list[tuple[Any, ...]]:
    """Capture actual states, all edges, completeness and unopened branches."""
    return [
        (
            n.id,
            id(n.state),
            n.tree_node.all_branches_generated,
            tuple(
                (b, c.id if c is not None else None) for b, c in n.iter_child_links()
            ),
            tuple((p.id, frozenset(bs)) for p, bs in n.parent_nodes.items()),
            frozenset(n.tree_node.non_opened_branches),
        )
        for n in nodes(search)
    ]


@pytest.mark.parametrize("batch_size", [1, 2, 64])
def test_repeated_refresh_dag_matches_independent_oracle(batch_size: int) -> None:
    search = runtime(_CHILDREN, _VALUES, steps=3)
    shared = next(n for n in nodes(search) if n.tag == 3)
    assert len(shared.parent_nodes) == 2
    before = topology(search)
    for score in (42.0, 7.0, 42.0):
        evaluator = Evaluator({i: score for i in _VALUES})
        expected = reference(search, evaluator)
        evaluator.batches.clear()
        report = search.refresh_with_evaluator(
            evaluator, scope="all", batch_size=batch_size
        )
        assert snapshot(search) == expected
        assert max(map(len, evaluator.batches)) <= batch_size
        assert report.requested_count == len(expected)
        assert all(
            n.tree_evaluation.direct_evaluation_version == search.evaluator_version
            for n in nodes(search)
            if n.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
        )
        assert topology(search) == before


def test_terminal_and_missing_direct_are_independent_of_effective_certainty() -> None:
    search = runtime({0: [1], 1: []}, {0: 2.0, 1: 35.0}, steps=1)
    root = search.tree.root_node
    child = next(root.iter_child_nodes())
    terminal, version = (
        child.tree_evaluation.direct_value,
        child.tree_evaluation.direct_evaluation_version,
    )
    root.tree_evaluation.clear_direct_evaluation()
    evaluator = Evaluator({0: 42.0, 1: 1000.0})
    expected = reference(search, evaluator)
    report = search.refresh_with_evaluator(evaluator, scope="all", batch_size=1)
    assert snapshot(search) == expected
    assert report.skipped_terminal_count == 1
    assert child.tree_evaluation.direct_value is terminal
    assert child.tree_evaluation.direct_evaluation_version == version
    assert root.tree_evaluation.direct_value == Value(
        score=42.0, certainty=Certainty.ESTIMATE
    )
    assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
    assert root.tree_evaluation.tree_value.score == 35.0


@pytest.mark.parametrize("scope", ["all", "root_children"])
def test_certified_direct_provenance_is_preserved(scope: str) -> None:
    search = runtime({0: [1], 1: [2], 2: []}, {0: 0.0, 1: 1.0, 2: 9.0}, steps=1)
    child = next(search.tree.root_node.iter_child_nodes())
    certificate = Value(score=9.0, certainty=Certainty.FORCED)
    child.tree_evaluation.direct_value = certificate
    version = child.tree_evaluation.direct_evaluation_version
    search.tree_manager.value_propagator.propagate_from_changed_nodes([child])
    evaluator = Evaluator({0: 42.0, 1: 42.0, 2: 42.0})
    report = search.refresh_with_evaluator(
        evaluator,
        scope=scope,
        validate_direct_certificate=lambda n: (
            n is child
            and n.tree_evaluation.direct_value is certificate
            and certifies(n, {2: 9.0})
        ),
    )
    assert child.tree_evaluation.direct_value is certificate
    assert child.tree_evaluation.direct_evaluation_version == version
    assert report.skipped_certified_count == 1
    assert all(child.id not in batch for batch in evaluator.batches)


def test_unverified_forced_label_refuses_before_changing_evaluator() -> None:
    search = runtime({0: [1], 1: []}, {0: 0.0, 1: 1.0})
    search.tree.root_node.tree_evaluation.direct_value = Value(
        score=9.0, certainty=Certainty.FORCED
    )
    old = search.tree_manager.node_evaluator.master_state_value_evaluator
    before = snapshot(search)
    with pytest.raises(ValueError, match="unverified direct FORCED"):
        search.refresh_with_evaluator(Evaluator({0: 42.0}), scope="all")
    assert search.evaluator_version == 0
    assert search.tree_manager.node_evaluator.master_state_value_evaluator is old
    assert snapshot(search) == before


@pytest.mark.parametrize("complete", [False, True])
def test_external_tree_proof_must_match_existing_backup_rule(complete: bool) -> None:
    search = runtime({0: [1], 1: [2], 2: []}, {0: 0.0, 1: 1.0, 2: 2.0}, steps=1)
    root = search.tree.root_node
    root.tree_node.all_branches_generated = complete
    certificate = Value(score=35.0, certainty=Certainty.FORCED)
    root.tree_evaluation.tree_value = certificate
    before = snapshot(search)
    evaluator = Evaluator({0: 42.0, 1: 42.0})
    with pytest.raises(ValueError, match="unverifiable tree certificate"):
        search.refresh_with_evaluator(evaluator, scope="all")
    assert search.evaluator_version == 0 and not evaluator.batches
    assert snapshot(search) == before


def test_direct_update_does_not_revoke_external_tree_certificate() -> None:
    search = runtime({0: [1], 1: [2], 2: []}, {0: 0.0, 1: 1.0, 2: 2.0}, steps=1)
    root = search.tree.root_node
    root.tree_node.all_branches_generated = False
    certificate = Value(score=35.0, certainty=Certainty.FORCED)
    root.tree_evaluation.tree_value = certificate
    search.apply_node_value_updates(
        [NodeValueUpdate(node_id=str(root.id), direct_value=42.0)]
    )
    assert root.tree_evaluation.tree_value is certificate
    assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE


def test_effective_only_change_reaches_both_parents_and_root() -> None:
    search = runtime(_CHILDREN, _VALUES, steps=3)
    shared = next(n for n in nodes(search) if n.tag == 3)
    shared.tree_node.all_branches_generated = False
    tree, pv = (
        shared.tree_evaluation.tree_value,
        tuple(shared.tree_evaluation.best_branch_sequence),
    )
    search.apply_node_value_updates(
        [NodeValueUpdate(node_id=str(shared.id), direct_value=100.0)]
    )
    assert shared.tree_evaluation.tree_value == tree
    assert tuple(shared.tree_evaluation.best_branch_sequence) == pv
    assert snapshot(search) == reference(search)
    assert all(
        p.tree_evaluation.effective_value.score == 100.0 for p in shared.parent_nodes
    )
    assert search.tree.root_node.tree_evaluation.effective_value.score == 100.0


def test_fully_opened_node_keeps_tree_preference_when_direct_changes() -> None:
    search = runtime(_CHILDREN, _VALUES, steps=3)
    shared = next(n for n in nodes(search) if n.tag == 3)
    assert shared.tree_node.all_branches_generated
    search.apply_node_value_updates(
        [NodeValueUpdate(node_id=str(shared.id), direct_value=100.0)]
    )
    assert shared.tree_evaluation.effective_value_source.value == "tree_child"
    assert snapshot(search) == reference(search)


def test_equal_scores_preserve_exactness_order_and_effective_source() -> None:
    search = runtime(
        {0: [1, 2], 1: [], 2: [3], 3: []}, {0: 0.0, 1: 9.0, 2: 1.0, 3: 0.0}, steps=1
    )
    root = search.tree.root_node
    children = list(root.iter_child_nodes())
    estimated = next(n for n in children if n.tag == 2)
    estimated.tree_evaluation.direct_value = Value(
        score=9.0, certainty=Certainty.ESTIMATE
    )
    search.tree_manager.value_propagator.propagate_from_changed_nodes([estimated])
    assert snapshot(search) == reference(search)
    assert root.tree_evaluation.decision_ordered_branches()[0] == next(
        b for b, c in root.iter_child_links() if c.tag == 1
    )
    root.tree_node.all_branches_generated = False
    root.tree_evaluation.direct_value = Value(score=9.0, certainty=Certainty.FORCED)
    certificate = root.tree_evaluation.direct_value
    evaluator = Evaluator({0: 42.0, 1: 99.0, 2: 9.0 / ((1 / 0.99999999) ** 1), 3: 0.0})
    expected = reference(search, evaluator)
    search.refresh_with_evaluator(
        evaluator,
        scope="all",
        validate_direct_certificate=lambda n: (
            n is root
            and n.tree_evaluation.direct_value is certificate
            and certifies(n, {1: 9.0, 3: 0.0})
        ),
    )
    assert snapshot(search) == expected
    assert root.tree_evaluation.effective_value is certificate
    assert root.tree_evaluation.tree_value.certainty == Certainty.ESTIMATE


def test_full_node_conflicting_direct_proof_is_refused() -> None:
    search = runtime({0: [1], 1: [2], 2: []}, {0: 0.0, 1: 1.0, 2: 9.0}, steps=1)
    root = search.tree.root_node
    root.tree_evaluation.direct_value = Value(score=9.0, certainty=Certainty.FORCED)
    with pytest.raises(ValueError, match="conflicts with tree/effective"):
        search.refresh_with_evaluator(
            Evaluator({0: 42.0, 1: 42.0}),
            scope="all",
            validate_direct_certificate=lambda n: certifies(n, {2: 9.0}),
        )
    assert search.evaluator_version == 0


@pytest.mark.parametrize("part", ["backup", "aggregation", "proof"])
def test_unknown_backup_proof_policies_fail_closed(part: str) -> None:
    search = runtime({0: [1], 1: [2], 2: []}, {0: 0.0, 1: 1.0, 2: 2.0}, steps=1)
    evaluation = search.tree.root_node.tree_evaluation
    if part == "backup":

        class Custom(ExplicitMaxBackupPolicy):
            """Unknown policy subclass deliberately excluded from refresh."""

        evaluation.backup_policy = Custom()
    else:
        setattr(evaluation.backup_policy, f"{part}_policy", object())
    with pytest.raises(ValueError, match="unsupported backup/proof"):
        search.refresh_with_evaluator(Evaluator({0: 42.0, 1: 42.0}), scope="all")
    assert search.evaluator_version == 0


def test_idempotent_predictions_still_settle_all_backups_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = runtime(_CHILDREN, _VALUES, steps=3)
    evaluator = Evaluator(_VALUES)
    before = snapshot(search)
    expected = reference(search, evaluator)
    calls = []
    selector_type = type(search.node_selector)
    invalidate = selector_type.invalidate
    refresh = selector_type.refresh_state_for_checkpoint

    def tracked_invalidate(selector: Any) -> None:
        calls.append("invalidate")
        invalidate(selector)

    def tracked_refresh(selector: Any, **kwargs: Any) -> None:
        calls.append("refresh")
        refresh(selector, **kwargs)

    monkeypatch.setattr(selector_type, "invalidate", tracked_invalidate)
    monkeypatch.setattr(selector_type, "refresh_state_for_checkpoint", tracked_refresh)
    report = search.refresh_with_evaluator(evaluator, scope="all", batch_size=1)
    assert snapshot(search) == expected == before
    assert report.changed_count == 0
    assert calls == ["invalidate", "refresh"]


def test_explicit_low_level_certificate_replacement_remains_available() -> None:
    search = runtime({0: [1], 1: []}, {0: 0.0, 1: 1.0})
    root = search.tree.root_node
    root.tree_evaluation.direct_value = Value(score=9.0, certainty=Certainty.FORCED)
    search.tree_manager.node_evaluator.master_state_value_evaluator = Evaluator(
        {0: 42.0}
    )
    search.tree_manager.node_evaluator.evaluate_nodes(
        [root],
        evaluation_queries=EvaluationQueries(),
        clear_existing_direct_values=True,
        skip_terminal_nodes=False,
    )
    assert root.tree_evaluation.direct_value == Value(
        score=42.0, certainty=Certainty.ESTIMATE
    )


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_batch_size_is_rejected_before_mutation(size: int) -> None:
    search = runtime({0: [1], 1: []}, {0: 0.0, 1: 1.0})
    with pytest.raises(ValueError, match="batch_size must be positive"):
        search.refresh_with_evaluator(
            Evaluator({0: 42.0}), scope="all", batch_size=size
        )
    assert search.evaluator_version == 0


def test_refreshed_checkpoint_preserves_rng_versions_and_next_selections() -> None:
    from random import Random

    from anemone.checkpoints import (
        build_search_checkpoint_payload,
        load_search_from_checkpoint_payload,
    )
    from anemone.node_evaluation.tree.single_agent.factory import (
        NodeMaxEvaluationFactory,
    )
    from tests.fake_yaml_game import FakeYamlDynamics, MasterStateValueEvaluatorFromYaml
    from tests.test_checkpoint_load import (
        _CHILDREN_BY_ID,
        _build_linoo_args,
        _build_linoo_runtime,
        _ConcreteFakeYamlState,
        _FakeIncrementalStateCheckpointCodec,
        _values_for,
    )

    search = _build_linoo_runtime()
    search.node_selector.base.depth_selection_policy = "alternating_by_step"
    for _ in range(2):
        search.step()
    evaluator = MasterStateValueEvaluatorFromYaml(
        {i: v * 2 for i, v in _values_for(_CHILDREN_BY_ID).items()}
    )
    selector = search.node_selector.base
    rng, step = selector.random_generator.getstate(), selector._selection_step_count
    search.refresh_with_evaluator(evaluator, scope="all", batch_size=1)
    assert snapshot(search) == reference(search)
    assert selector.random_generator.getstate() == rng
    assert selector._selection_step_count == step
    codec = _FakeIncrementalStateCheckpointCodec(_CHILDREN_BY_ID)
    payload = build_search_checkpoint_payload(search, state_codec=codec)
    args = _build_linoo_args()
    args.node_selector.base.depth_selection_policy = "alternating_by_step"
    restored = load_search_from_checkpoint_payload(
        payload,
        state_codec=codec,
        dynamics=FakeYamlDynamics(),
        args=args,
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=evaluator,
        random_generator=Random(999),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    assert snapshot(restored) == snapshot(search)
    assert restored.evaluator_version == search.evaluator_version
    assert [
        (n.id, n.tree_evaluation.direct_evaluation_version) for n in nodes(restored)
    ] == [(n.id, n.tree_evaluation.direct_evaluation_version) for n in nodes(search)]
    for _ in range(3):
        search.step()
        restored.step()
        assert (
            search.latest_step_report.selected_node_id
            == restored.latest_step_report.selected_node_id
        )
        assert (
            selector.random_generator.getstate()
            == restored.node_selector.base.random_generator.getstate()
        )
        assert (
            selector._selection_step_count
            == restored.node_selector.base._selection_step_count
        )
        assert snapshot(search) == snapshot(restored)


def test_nonterminal_terminal_label_is_refused() -> None:
    from valanga import Outcome, OverEvent

    search = runtime({0: [1], 1: []}, {0: 0.0, 1: 1.0})
    search.tree.root_node.tree_evaluation.direct_value = Value(
        score=9.0,
        certainty=Certainty.TERMINAL,
        over_event=OverEvent(outcome=Outcome.WIN, winner=None, termination="forged"),
    )
    with pytest.raises(ValueError, match="TERMINAL direct value on a nonterminal"):
        search.refresh_with_evaluator(Evaluator({0: 42.0}), scope="all")
    assert search.evaluator_version == 0


def test_equal_score_direct_certificate_changes_effective_selection_in_dag() -> None:
    search = runtime(
        {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: [5], 5: []},
        {i: 9.0 if i == 5 else 1.0 for i in range(6)},
        steps=3,
    )
    by_tag = {n.tag: n for n in nodes(search)}
    shared, leaf = by_tag[3], by_tag[4]
    shared.tree_node.all_branches_generated = False
    shared.tree_evaluation.direct_value = Value(score=9.0, certainty=Certainty.ESTIMATE)
    leaf.tree_evaluation.direct_value = Value(score=9.0, certainty=Certainty.ESTIMATE)
    search.tree_manager.value_propagator.propagate_from_changed_nodes([leaf])
    before = shared.tree_evaluation.tree_value
    assert shared.tree_evaluation.effective_value_source.value == "tree_child"
    assert shared.tree_evaluation.effective_value.score == 9.0
    search.apply_node_value_updates(
        [NodeValueUpdate(node_id=str(shared.id), direct_value=9.0, is_exact=True)]
    )
    assert shared.tree_evaluation.tree_value == before
    assert shared.tree_evaluation.effective_value_source.value == "direct_self"
    assert certifies(shared, {5: 9.0})
    assert shared.tree_evaluation.effective_value.certainty == Certainty.FORCED
    assert all(
        p.tree_evaluation.tree_value.certainty == Certainty.FORCED
        for p in shared.parent_nodes
    )
    assert (
        search.tree.root_node.tree_evaluation.tree_value.certainty == Certainty.FORCED
    )
    assert snapshot(search) == reference(search)


def test_direct_update_cannot_silently_revoke_an_ancestor_tree_proof() -> None:
    search = runtime({0: [1], 1: [2], 2: []}, {0: 0.0, 1: 1.0, 2: 35.0}, steps=1)
    root = search.tree.root_node
    child = next(root.iter_child_nodes())
    child.tree_evaluation.direct_value = Value(score=35.0, certainty=Certainty.FORCED)
    search.tree_manager.value_propagator.propagate_from_changed_nodes([child])
    certificate = root.tree_evaluation.tree_value
    assert certificate.certainty == Certainty.FORCED
    with pytest.raises(ValueError, match="unverifiable tree certificate"):
        search.apply_node_value_updates(
            [NodeValueUpdate(node_id=str(child.id), direct_value=42.0, is_exact=False)]
        )
    assert root.tree_evaluation.tree_value is certificate
