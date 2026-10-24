"""Ordered real Morpion continuation through repeated first-generation checkpoints."""
# ruff: noqa: D103

import logging
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Any

import pytest
from atomheart.games.morpion import initial_state
from valanga.evaluations import Certainty, Value

from anemone import create_search_with_tree_eval_factory
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    checkpoint_payload_to_jsonable,
    load_checkpoint_json_payload,
    load_search_from_checkpoint_payload,
    load_search_from_sharded_checkpoint,
    write_checkpoint_json_payload,
    write_sharded_search_checkpoint,
)
from anemone.checkpoints.rng_state import rollout_random_generator
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.tree_manager.opening_expansion_config import (
    OpeningExpansionConfig,
    OpeningExpansionKind,
    RolloutActionSelectorKind,
    RolloutExpansionConfig,
)
from tests.morpion_checkpoint_cache_helpers import Codec
from tests.morpion_refresh_helpers import Dynamics, Evaluator, State
from tests.scientific_refresh_helpers import nodes, snapshot
from tests.test_checkpoint_load import _build_linoo_args
from tests.test_checkpoint_ordered_continuation import ordered
from tests.test_checkpoint_rollout_rng import _decode_json
from tests.test_linoo_alternating import _signature


class ConstantEvaluator(Evaluator):
    """Deterministic nonterminal prediction, with genuine local terminal detection."""

    def evaluate_batch_items(self, items: Any) -> list[Value]:
        """Keep the same prediction for future newly created nodes too."""
        return [Value(score=42.0, certainty=Certainty.ESTIMATE) for _ in items]


def make_runtime(rollout: bool) -> tuple[Any, Any]:
    """Use actual rules, default-max/Linoo, and independent configured rollout RNG."""
    args = _build_linoo_args()
    args.stopping_criterion.tree_branch_limit = 100000
    args.node_selector.base = replace(
        args.node_selector.base, depth_selection_policy="alternating_by_step"
    )
    if rollout:
        args.opening_expansion = OpeningExpansionConfig(
            kind=OpeningExpansionKind.ROLLOUT,
            rollout=RolloutExpansionConfig(
                max_extra_steps=1,
                action_selector_kind=RolloutActionSelectorKind.RANDOM_LEGAL_PREFER_OPENABLE,
                random_seed=0,
                stop_on_existing_node=False,
            ),
        )
    search = create_search_with_tree_eval_factory(
        state_type=State,
        dynamics=Dynamics(),
        starting_state=State(initial_state()),
        args=args,
        random_generator=Random(4),
        master_state_evaluator=Evaluator(),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    while search.tree.nodes_count < 1000:
        search.step()
    return search, args


def payload(search: Any, codec: Any) -> Any:
    """Capture the complete checkpoint, with no normalized-away fields."""
    return checkpoint_payload_to_jsonable(
        build_search_checkpoint_payload(search, state_codec=codec)
    )


def restore(
    search: Any, args: Any, codec: Any, storage: str, path: Path, evaluator: Any
) -> Any:
    """Restore with the identical evaluator supplied directly at the loader boundary."""
    exported = build_search_checkpoint_payload(search, state_codec=codec)
    path.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        state_codec=codec,
        dynamics=Dynamics(),
        args=args,
        state_type=State,
        master_state_value_evaluator=evaluator,
        random_generator=Random(999),
        node_tree_evaluation_factory=NodeMaxEvaluationFactory(),
    )
    if storage == "json-zst":
        filename = path / "checkpoint.json.zst"
        write_checkpoint_json_payload(exported, filename)
        return load_search_from_checkpoint_payload(
            _decode_json(load_checkpoint_json_payload(filename)[0]), **kwargs
        )
    write_sharded_search_checkpoint(exported, path, node_count_per_shard=97)
    return load_search_from_sharded_checkpoint(path, **kwargs)


def semantic_report(search: Any) -> Any:
    """Compare decisions/expansions; elapsed time and cache work are diagnostics."""
    report = search.latest_step_report
    selection = search.node_selector.base.latest_selection_report
    return (
        report.selected_node_id,
        report.selected_depth,
        report.nodes_before,
        report.nodes_after,
        report.nodes_added,
        report.branch_count,
        report.selector_report_rows,
        _signature(selection),
        (
            selection.selected_node_direct_value,
            selection.selected_node_candidate_value,
            selection.selected_node_priority,
            selection.selected_node_rank,
            selection.ranked_candidate_count,
            selection.node_selection_policy,
            selection.depth_selection_policy,
        ),
        getattr(search.tree_manager.opening_expansion_executor, "last_report", None),
        [
            (
                e.parent_node.id if e.parent_node else None,
                e.branch_key,
                e.child_node.id,
                e.creation_child_node,
            )
            for e in search.latest_tree_expansions
        ],
    )


@pytest.mark.parametrize("storage", ["json-zst", "sharded"])
@pytest.mark.parametrize("rollout", [False, True])
def test_actual_1k_repeated_checkpoints_through_30_growth_steps(
    storage: str, rollout: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anemone.utils.logger import anemone_logger

    monkeypatch.setattr(anemone_logger, "handlers", [logging.StreamHandler()])
    live, args = make_runtime(rollout)
    codec = Codec()
    continued = restore(live, args, codec, storage, tmp_path / "initial", Evaluator())
    assert ordered(live) == ordered(continued)
    assert payload(live, codec) == payload(continued, codec)
    assert [n.state.board.canonical_tag for n in nodes(live)] == [
        n.state.board.canonical_tag for n in nodes(continued)
    ]
    evaluator = ConstantEvaluator()
    assert live.refresh_with_evaluator(
        evaluator, scope="all", batch_size=64
    ) == continued.refresh_with_evaluator(evaluator, scope="all", batch_size=64)
    assert payload(live, codec) == payload(continued, codec)
    for step in range(30):
        live.step()
        continued.step()
        assert semantic_report(live) == semantic_report(continued)
        assert snapshot(live) == snapshot(continued)
        assert ordered(live) == ordered(continued)
        assert (
            live.node_selector.base.random_generator.getstate()
            == continued.node_selector.base.random_generator.getstate()
        )
        lr, rr = rollout_random_generator(live), rollout_random_generator(continued)
        assert (None if lr is None else lr.getstate()) == (
            None if rr is None else rr.getstate()
        )
        if step % 5 == 4:
            before = payload(live, codec)
            assert payload(continued, codec) == before
            continued = restore(
                continued,
                args,
                codec,
                storage,
                tmp_path / f"step-{step + 1}",
                evaluator,
            )
            assert ordered(live) == ordered(continued)
            assert payload(continued, codec) == before
    assert [n.state.board.canonical_tag for n in nodes(live)] == [
        n.state.board.canonical_tag for n in nodes(continued)
    ]
