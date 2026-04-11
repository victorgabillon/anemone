"""Tests for exporting live search runtimes into checkpoint payloads."""

from __future__ import annotations

from random import Random
from typing import Any

import pytest
from valanga import Color

from anemone import SearchArgs, create_search
from anemone.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    LinkedChildCheckpointPayload,
    SearchRuntimeCheckpointPayload,
    TreeExpansionsCheckpointPayload,
    build_search_checkpoint_payload,
)
from anemone.checkpoints.value_serialization import serialize_checkpoint_atom
from anemone.indices.node_indices.index_types import IndexComputationType
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.uniform.uniform import UniformArgs
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeBranchLimitArgs,
)
from anemone.recommender_rule.recommender_rule import SoftmaxRule
from tests.fake_yaml_game import (
    FakeYamlDynamics,
    FakeYamlState,
    MasterStateValueEvaluatorFromYaml,
)


class _ConcreteFakeYamlState(FakeYamlState):
    """Concrete test state satisfying the abstract ``State`` surface."""

    def pprint(self) -> str:
        """Return a small human-readable representation for test diagnostics."""
        return str(self.node_id)


class _FakeStateCheckpointCodec:
    """Tiny external-state-ref codec for checkpoint exporter tests."""

    def dump_state_ref(self, state: _ConcreteFakeYamlState) -> object:
        """Return an opaque state reference produced outside Anemone."""
        return {
            "codec": "fake-yaml",
            "tag": state.tag,
            "node_id": state.node_id,
            "turn": state.turn.name,
        }


_CHILDREN_BY_ID = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [],
    4: [],
}

_VALUE_BY_ID = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 4.0,
}


def _build_args(
    *,
    index_computation: IndexComputationType | None = None,
) -> SearchArgs:
    """Return a deterministic search configuration for live-export tests."""
    return SearchArgs(
        node_selector=ComposedNodeSelectorArgs(
            type=NodeSelectorType.COMPOSED,
            priority=NoPriorityCheckArgs(type=NodeSelectorType.PRIORITY_NOOP),
            base=UniformArgs(type=NodeSelectorType.UNIFORM),
        ),
        opening_type=OpeningType.ALL_CHILDREN,
        stopping_criterion=TreeBranchLimitArgs(
            type=StoppingCriterionTypes.TREE_BRANCH_LIMIT,
            tree_branch_limit=10,
        ),
        recommender_rule=SoftmaxRule(type="softmax", temperature=1.0),
        index_computation=index_computation,
    )


def _build_runtime(
    *,
    index_computation: IndexComputationType | None = None,
) -> Any:
    """Build one small real search runtime over the fake YAML domain."""
    starting_state = _ConcreteFakeYamlState(
        node_id=0,
        children_by_id=_CHILDREN_BY_ID,
        turn=Color.WHITE,
    )
    return create_search(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        starting_state=starting_state,
        args=_build_args(index_computation=index_computation),
        random_generator=Random(0),
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(_VALUE_BY_ID),
        state_representation_factory=None,
    )


def _build_payload(runtime: Any) -> SearchRuntimeCheckpointPayload:
    """Export a runtime using the fake external state-ref codec."""
    return build_search_checkpoint_payload(
        runtime,
        state_codec=_FakeStateCheckpointCodec(),
    )


def _nodes_by_id(payload: Any) -> dict[int, Any]:
    """Return checkpoint node payloads keyed by node id."""
    return {node.node_id: node for node in payload.tree.nodes}


def test_build_search_checkpoint_payload_from_live_search() -> None:
    """A live runtime should export top-level and tree checkpoint metadata."""
    runtime = _build_runtime()
    runtime.step()
    runtime.step()

    payload = _build_payload(runtime)

    assert payload.format_version == CHECKPOINT_FORMAT_VERSION
    assert payload.evaluator_version == runtime.evaluator_version
    assert payload.tree.root_node_id == runtime.tree.root_node.id
    assert len(payload.tree.nodes) == runtime.tree.nodes_count
    assert len(payload.tree.nodes) == runtime.tree.descendants.get_count()
    assert isinstance(payload.latest_tree_expansions, TreeExpansionsCheckpointPayload)


def test_node_state_refs_come_from_external_codec() -> None:
    """Exported node state refs should be exactly the codec-produced objects."""
    runtime = _build_runtime()
    runtime.step()

    payload = _build_payload(runtime)

    assert {
        node_payload.node_id: node_payload.state_ref
        for node_payload in payload.tree.nodes
    } == {
        node.id: _FakeStateCheckpointCodec().dump_state_ref(node.state)
        for tree_depth in runtime.tree.descendants.range()
        for node in runtime.tree.descendants[tree_depth].values()
    }


def test_direct_and_backed_up_values_are_serialized_from_live_nodes() -> None:
    """The exporter should preserve direct and backed-up values already present."""
    runtime = _build_runtime()
    runtime.step()
    root = runtime.tree.root_node
    payload = _build_payload(runtime)
    root_payload = _nodes_by_id(payload)[root.id]

    assert root_payload.evaluation is not None
    assert root_payload.evaluation.direct_value is not None
    assert root_payload.evaluation.direct_value.score == pytest.approx(
        root.tree_evaluation.direct_value.score
    )
    assert root_payload.evaluation.direct_evaluation_version == 0
    assert root_payload.evaluation.backed_up_value is not None
    assert root_payload.evaluation.backed_up_value.score == pytest.approx(
        root.tree_evaluation.backed_up_value.score
    )

    for child in root.branches_children.values():
        assert child is not None
        child_payload = _nodes_by_id(payload)[child.id]
        assert child_payload.evaluation is not None
        assert child_payload.evaluation.direct_value is not None
        assert child_payload.evaluation.direct_evaluation_version == 0


def test_structural_parent_and_child_relationships_are_exported() -> None:
    """Node payloads should preserve parent ids and child branch mappings."""
    runtime = _build_runtime()
    runtime.step()
    root = runtime.tree.root_node
    payload = _build_payload(runtime)
    payload_nodes = _nodes_by_id(payload)
    root_payload = payload_nodes[root.id]

    assert root_payload.parent_node_id is None
    assert root_payload.branch_from_parent is None
    assert root_payload.linked_children == sorted(
        [
            LinkedChildCheckpointPayload(
                branch_key=serialize_checkpoint_atom(branch),
                child_node_id=child.id,
            )
            for branch, child in root.branches_children.items()
            if child is not None
        ],
        key=lambda item: (repr(item.branch_key), item.child_node_id),
    )

    for branch, child in root.branches_children.items():
        assert child is not None
        child_payload = payload_nodes[child.id]
        assert child_payload.parent_node_id == root.id
        assert child_payload.branch_from_parent == serialize_checkpoint_atom(branch)


def test_evaluation_runtime_caches_are_exported_when_present() -> None:
    """PV/frontier/ordering/backup-runtime cache payloads should reflect live state."""
    runtime = _build_runtime()
    runtime.step()
    root = runtime.tree.root_node
    root_payload = _nodes_by_id(_build_payload(runtime))[root.id]

    assert root_payload.evaluation is not None
    evaluation_payload = root_payload.evaluation
    assert evaluation_payload.principal_variation is not None
    assert evaluation_payload.principal_variation.pv_version == (
        root.tree_evaluation.pv_state.pv_version
    )
    assert evaluation_payload.principal_variation.best_branch_sequence == [
        serialize_checkpoint_atom(branch)
        for branch in root.tree_evaluation.pv_state.best_branch_sequence
    ]
    assert evaluation_payload.branch_frontier is not None
    assert set(evaluation_payload.branch_frontier.frontier_branches) == {
        serialize_checkpoint_atom(branch)
        for branch in root.tree_evaluation.branch_frontier.frontier_branches
    }
    assert evaluation_payload.decision_ordering is not None
    assert len(evaluation_payload.decision_ordering.branch_ordering) == len(
        root.tree_evaluation.decision_ordering.branch_ordering_keys
    )
    assert evaluation_payload.backup_runtime is not None
    assert evaluation_payload.backup_runtime.is_initialized is True
    assert evaluation_payload.backup_runtime.best_branch == serialize_checkpoint_atom(
        root.tree_evaluation.backup_runtime.best_branch
    )


def test_exploration_index_payload_is_exported_when_present() -> None:
    """Exploration-index data should export stable pointer-free fields."""
    runtime = _build_runtime(index_computation=IndexComputationType.RECUR_ZIPF)
    runtime.step()

    root_payload = _nodes_by_id(_build_payload(runtime))[runtime.tree.root_node.id]

    assert root_payload.exploration_index is not None
    assert root_payload.exploration_index.kind.endswith("ExplorationData")
    assert isinstance(root_payload.exploration_index.payload, dict)
    assert "tree_node" not in root_payload.exploration_index.payload
    assert "index" in root_payload.exploration_index.payload


def test_minimal_fresh_runtime_can_be_exported() -> None:
    """A newly created runtime should export a valid root-only checkpoint."""
    runtime = _build_runtime()

    payload = _build_payload(runtime)

    assert payload.tree.root_node_id == runtime.tree.root_node.id
    assert len(payload.tree.nodes) == 1
    assert payload.tree.nodes[0].node_id == runtime.tree.root_node.id
    assert payload.tree.nodes[0].evaluation is not None
    assert payload.tree.nodes[0].evaluation.direct_value is not None
    assert payload.latest_tree_expansions is not None
    assert len(payload.latest_tree_expansions.expansions_with_node_creation) == 1
    root_expansion = payload.latest_tree_expansions.expansions_with_node_creation[0]
    assert root_expansion.child_node_id == runtime.tree.root_node.id
    assert root_expansion.parent_node_id is None
    assert root_expansion.branch_key is None
