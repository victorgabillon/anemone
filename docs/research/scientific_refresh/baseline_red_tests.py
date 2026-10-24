"""Baseline-only red witnesses; run against approved a07fb2f source, before fixes."""
# ruff: noqa: D103

from valanga.evaluations import Certainty, Value

from anemone import NodeValueUpdate
from tests.morpion_refresh_helpers import Evaluator, runtime
from tests.scientific_refresh_helpers import Evaluator as FixtureEvaluator
from tests.scientific_refresh_helpers import certifies
from tests.scientific_refresh_helpers import runtime as fixture_runtime


def test_valid_forced_nine_is_preserved() -> None:
    search = fixture_runtime({0: [1], 1: []}, {0: 0.0, 1: 9.0})
    root = search.tree.root_node
    certificate = Value(score=9.0, certainty=Certainty.FORCED)
    root.tree_evaluation.direct_value = certificate
    assert certifies(root, {1: 9.0})  # Unique legal continuation ends at nine.
    search.refresh_with_evaluator(FixtureEvaluator({0: 42.0, 1: 42.0}), scope="all")
    assert root.tree_evaluation.direct_value is certificate


def test_seven_node_full_refresh_reaches_root() -> None:
    search = runtime()
    assert search.tree.nodes_count == 7
    search.refresh_with_evaluator(Evaluator({2: 100.0}), scope="all")
    assert (
        search.tree.root_node.tree_evaluation.effective_value.score
        == 100.00000100000001
    )


def test_seven_node_incremental_update_reaches_root() -> None:
    search = runtime()
    search.apply_node_value_updates(
        [NodeValueUpdate(node_id="2", direct_value=100.00000100000001)]
    )
    assert (
        search.tree.root_node.tree_evaluation.effective_value.score
        == 100.00000100000001
    )
