"""Independent regressions for model refresh and effective-value propagation."""
# ruff: noqa: D103

import pytest
from valanga.evaluations import Certainty, Value

from anemone import NodeValueUpdate
from tests.morpion_refresh_helpers import Evaluator, runtime
from tests.scientific_refresh_helpers import Evaluator as FixtureEvaluator
from tests.scientific_refresh_helpers import certifies, reference, snapshot
from tests.scientific_refresh_helpers import runtime as fixture_runtime


def test_forced_direct_certificate_survives_model_refresh() -> None:
    search = fixture_runtime({0: [1], 1: []}, {0: 0.0, 1: 9.0})
    root = search.tree.root_node
    certificate = Value(score=9, certainty=Certainty.FORCED)
    root.tree_evaluation.direct_value = certificate
    version = root.tree_evaluation.direct_evaluation_version
    search.refresh_with_evaluator(
        FixtureEvaluator({0: 42.0, 1: 42.0}),
        scope="all",
        validate_direct_certificate=lambda node: (
            node is root
            and node.tree_evaluation.direct_value is certificate
            and certifies(node, {1: 9.0})
        ),
    )
    assert root.tree_evaluation.direct_value is certificate
    assert root.tree_evaluation.direct_evaluation_version == version


def test_actual_seven_node_morpion_full_refresh() -> None:
    search = runtime()
    nodes = {n.id: n for _, _, n in search.tree.descendants.iter_on_all_nodes()}
    assert len(nodes) == 7
    assert {
        n.id: [(b, c.id) for b, c in n.iter_child_links()]
        for n in nodes.values()
        if n.has_child_links()
    } == {
        0: [
            ((2, -5, 0, 2), 1),
            ((0, -6, -2, 0), 2),
            ((0, -5, -2, 4), 3),
            ((0, -3, -5, 0), 4),
        ],
        2: [((3, 0, 4, 2), 5), ((3, -5, -1, 2), 6)],
    }
    assert nodes[2].has_child_links() and not nodes[2].tree_node.all_branches_generated
    original_tree = nodes[2].tree_evaluation.tree_value
    assert (
        search.tree.root_node.tree_evaluation.effective_value.score
        == 2.2600000452000004
    )
    evaluator = Evaluator({2: 100})
    expected = reference(search, evaluator)
    search.refresh_with_evaluator(evaluator, scope="all")
    assert snapshot(search) == expected
    assert nodes[2].tree_evaluation.tree_value == original_tree
    assert (
        search.tree.root_node.tree_evaluation.effective_value.score
        == 100.00000100000001
    )


def test_actual_seven_node_morpion_incremental_update() -> None:
    search = runtime()
    changed = next(
        n for _, _, n in search.tree.descendants.iter_on_all_nodes() if n.id == 2
    )
    original_tree = changed.tree_evaluation.tree_value
    original_pv = tuple(changed.tree_evaluation.best_branch_sequence)
    search.apply_node_value_updates(
        [NodeValueUpdate(node_id="2", direct_value=100.00000100000001)]
    )
    assert changed.tree_evaluation.tree_value == original_tree
    assert tuple(changed.tree_evaluation.best_branch_sequence) == original_pv
    assert (
        search.tree.root_node.tree_evaluation.effective_value.score
        == 100.00000100000001
    )


def test_actual_endgame_preserves_forced_35_estimated_42() -> None:
    from atomheart.games.morpion import MorpionDynamics, initial_state

    from tests.morpion_refresh_helpers import State

    dynamics = MorpionDynamics()
    state = initial_state()
    for _ in range(200):
        actions = dynamics.all_legal_actions(state)
        assert actions
        result = dynamics.step(state, actions[0])
        if result.is_over:
            break
        state = result.next_state
    else:
        pytest.fail("endgame did not terminate")
    assert state.moves == 34 and result.next_state.moves == 35
    assert len(actions) == 1
    search = runtime(state=State(state), branch_limit=128, steps=1)
    root = search.tree.root_node
    terminal = next(root.iter_child_nodes())
    terminal_value = terminal.tree_evaluation.direct_value
    evaluator = Evaluator({0: 42})
    expected = reference(search, evaluator)
    search.refresh_with_evaluator(evaluator, scope="all", batch_size=1)
    assert snapshot(search) == expected
    assert root.tree_evaluation.direct_value.score == 42
    assert root.tree_evaluation.direct_value.certainty == Certainty.ESTIMATE
    assert root.tree_evaluation.tree_value.score == 35
    assert root.tree_evaluation.tree_value.certainty == Certainty.FORCED
    assert terminal.tree_evaluation.direct_value is terminal_value
    assert terminal_value.certainty == Certainty.TERMINAL


@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_real_dag_repeated_models_preserve_rng_search_and_topology(
    batch_size: int,
) -> None:
    from tests.scientific_refresh_helpers import nodes
    from tests.test_evaluator_refresh_contract import topology

    search = runtime(branch_limit=100000, steps=0)
    while search.tree.nodes_count < 76:
        search.step()
    assert sum(len(n.parent_nodes) > 1 for n in nodes(search)) >= 7
    before = topology(search)
    selector = search.node_selector.base
    rng, step = selector.random_generator.getstate(), selector._selection_step_count
    expansions, latest_step = search.latest_tree_expansions, search.latest_step_report
    for prediction in (42.0, 7.0, 42.0):
        evaluator = Evaluator({n.id: prediction for n in nodes(search)})
        expected = reference(search, evaluator)
        evaluator.batches.clear()
        search.refresh_with_evaluator(evaluator, scope="all", batch_size=batch_size)
        assert snapshot(search) == expected
        assert topology(search) == before
        assert selector.random_generator.getstate() == rng
        assert selector._selection_step_count == step
        assert search.latest_tree_expansions is expansions
        assert search.latest_step_report is latest_step
        assert max(map(len, evaluator.batches)) <= batch_size
