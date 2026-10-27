"""Synthetic Linoo restore regression with real lazy checkpoint state handles."""

from random import Random
from time import perf_counter
from typing import Any

import pytest
from valanga import Color, Outcome, OverEvent
from valanga.evaluations import Certainty, Value

from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    DeltaCheckpointStatePayload,
)
from anemone.checkpoints.state_handles import (
    CheckpointBackedStateHandle,
    CheckpointStateResolver,
)
from anemone.node_evaluation.tree.node_tree_evaluation import NodeTreeEvaluationState
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.node_selector.linoo import Linoo
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.nodes.state_handles import MaterializedStateHandle
from anemone.nodes.tree_node import TreeNode
from anemone.objectives import SingleAgentMaxObjective
from anemone.tree_manager.tree_expander import TreeExpansions
from anemone.trees.descendants import RangedDescendants
from anemone.trees.tree import Tree
from tests.test_checkpoint_load import (
    _ConcreteFakeYamlState,
    _FakeIncrementalStateCheckpointCodec,
)
from tests.test_linoo_selector import _FakeOpeningInstructor


def _checkpoint_fixture() -> tuple[Any, Any, Any, Any, Any]:
    """Build a three-node tree with one partially opened, backed-up frontier."""
    children = {0: [1], 1: [2, 3], 2: [], 3: []}
    codec = _FakeIncrementalStateCheckpointCodec(children)
    objective = SingleAgentMaxObjective()
    states = [
        _ConcreteFakeYamlState(node_id=i, children_by_id=children, turn=Color.WHITE)
        for i in range(3)
    ]
    payloads = {
        0: AnchorCheckpointStatePayload(anchor_ref=codec.dump_anchor_ref(states[0]))
    }
    for i in (1, 2):
        payloads[i] = DeltaCheckpointStatePayload(
            state_parent_node_id=i - 1,
            state_parent_branch=0,
            delta_ref=codec.dump_delta_from_parent(
                parent_state=states[i - 1], child_state=states[i], branch_from_parent=0
            ),
        )
    resolver = CheckpointStateResolver(
        state_codec=codec, state_payloads_by_node_id=payloads
    )
    nodes = []
    descendants = RangedDescendants()
    for i, state in enumerate(states):
        structural = TreeNode(
            id_=i,
            tree_depth_=i,
            state_handle_=MaterializedStateHandle(state),
            parent_nodes_=None if i == 0 else (nodes[i - 1], 0),
            all_branches_generated=i != 1,
            non_opened_branches_={1} if i == 1 else set(),
        )
        evaluation = NodeMaxEvaluation(tree_node=structural, objective=objective)
        evaluation.direct_value = Value(2.0, Certainty.ESTIMATE)
        evaluation.backed_up_value = Value(2.0 if i == 0 else 1.0, Certainty.ESTIMATE)
        if i == 2:
            evaluation.direct_value = Value(
                1.0, Certainty.TERMINAL, OverEvent(outcome=Outcome.DRAW)
            )
            evaluation.backed_up_value = None
        node = AlgorithmNode(structural, evaluation, None, None)
        if nodes:
            nodes[-1].tree_node.set_child_for_branch(0, node)
        nodes.append(node)
        descendants.add_descendant(node)
    tree = Tree(root_node=nodes[0], descendants=descendants)
    tree.nodes_count = len(nodes)
    tree.branch_count = len(nodes) - 1
    source = Linoo(
        opening_instructor=_FakeOpeningInstructor(), random_generator=Random(0)
    )
    source.choose_node_and_branch_to_open(tree, TreeExpansions())
    payload = source.build_checkpoint_payload(objective)
    assert payload is not None
    for node in nodes:
        node.tree_node.state_handle_ = CheckpointBackedStateHandle(resolver, node.id)
    codec.reset_counters()
    return tree, objective, payload, resolver, codec


@pytest.mark.parametrize(
    "state_free", [False, True], ids=["stateful-control", "state-free"]
)
def test_linoo_restore_avoids_comparison_state_decodes(
    monkeypatch: pytest.MonkeyPatch, state_free: bool
) -> None:
    """Keep classifications and selector payload identical while eliminating loads."""
    tree, objective, payload, resolver, codec = _checkpoint_fixture()
    if not state_free:
        # This is exactly the previous dispatch: no optional callable is available.
        monkeypatch.setattr(
            SingleAgentMaxObjective,
            "semantic_compare_without_state",
            None,
            raising=False,
        )
    comparisons = []
    get_calls = []
    resolve_calls = []
    original_compare = NodeTreeEvaluationState.compare_candidate_values
    original_get = CheckpointBackedStateHandle.get
    original_resolve = CheckpointStateResolver.resolve

    def compare(evaluation: Any, left: Value, right: Value) -> int:
        comparisons.append((evaluation.tree_node.id, left, right))
        return original_compare(evaluation, left, right)

    def get(handle: Any) -> Any:
        get_calls.append(handle.node_id)
        return original_get(handle)

    def resolve(current_resolver: Any, node_id: int) -> Any:
        resolve_calls.append(node_id)
        return original_resolve(current_resolver, node_id)

    monkeypatch.setattr(NodeTreeEvaluationState, "compare_candidate_values", compare)
    monkeypatch.setattr(CheckpointBackedStateHandle, "get", get)
    monkeypatch.setattr(CheckpointStateResolver, "resolve", resolve)
    selector = Linoo(
        opening_instructor=_FakeOpeningInstructor(), random_generator=Random(0)
    )
    rng_before = selector.random_generator.getstate()
    started = perf_counter()
    assert selector.restore_from_checkpoint_payload(
        tree=tree, objective=objective, payload=payload
    )
    elapsed = perf_counter() - started

    # Terminality, exactness, effective-value classification, and heap restoration
    # each compare the direct and backed-up values of the partial frontier node.
    assert (
        comparisons
        == [(1, Value(2.0, Certainty.ESTIMATE), Value(1.0, Certainty.ESTIMATE))] * 4
    )
    assert selector.random_generator.getstate() == rng_before
    assert selector.build_checkpoint_payload(objective) == payload
    assert get_calls == ([] if state_free else [1] * 4)
    assert resolve_calls == ([] if state_free else [1, 0, 1, 1, 1])
    assert codec.loaded_anchor_node_ids == ([] if state_free else [0])
    assert codec.applied_delta_items == ([] if state_free else [(0, 1, 0)])
    assert set(resolver._resolved_states) == (set() if state_free else {0, 1})
    print(
        f"state_free={state_free} comparisons={len(comparisons)} "
        f"get_calls={len(get_calls)} resolve_calls={len(resolve_calls)} "
        f"decodes={len(codec.loaded_anchor_node_ids) + len(codec.applied_delta_items)} "
        f"resolved_states={len(resolver._resolved_states)} elapsed_s={elapsed:.6f}"
    )
