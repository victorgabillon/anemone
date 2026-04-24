"""Tests for exporting live search runtimes into checkpoint payloads."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any, cast

import pytest
from valanga import Color

from anemone import SearchArgs, create_search
from anemone.checkpoints import (
    AnchorCheckpointStatePayload,
    CHECKPOINT_FORMAT_VERSION,
    DeltaCheckpointStatePayload,
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


@dataclass(frozen=True, slots=True)
class _FakeCheckpointStateSummary:
    """Small typed summary object used by fake incremental codecs."""

    tag: int
    node_id: int


class _FakeIncrementalStateCheckpointCodec:
    """Tiny incremental checkpoint codec for exporter tests."""

    def __init__(self) -> None:
        """Initialize the codec call logs."""
        self.dumped_anchor_node_ids: list[int] = []
        self.dumped_delta_items: list[tuple[int, int, object | None]] = []
        self.dumped_summary_node_ids: list[int] = []

    def dump_anchor_ref(self, state: _ConcreteFakeYamlState) -> object:
        """Return an opaque anchor snapshot produced outside Anemone."""
        self.dumped_anchor_node_ids.append(state.node_id)
        return {
            "kind": "anchor",
            "node_id": state.node_id,
            "turn": state.turn.name,
        }

    def dump_delta_from_parent(
        self,
        *,
        parent_state: _ConcreteFakeYamlState,
        child_state: _ConcreteFakeYamlState,
        branch_from_parent: object | None = None,
    ) -> object:
        """Return an opaque delta from one concrete parent state."""
        self.dumped_delta_items.append(
            (parent_state.node_id, child_state.node_id, branch_from_parent)
        )
        return {
            "kind": "delta",
            "parent_node_id": parent_state.node_id,
            "child_node_id": child_state.node_id,
            "branch_from_parent": branch_from_parent,
            "turn": child_state.turn.name,
        }

    def load_anchor_ref(self, anchor_ref: object) -> _ConcreteFakeYamlState:
        """Exporter tests do not restore anchors."""
        raise AssertionError(f"Exporter tests should not load anchors: {anchor_ref!r}")

    def load_child_from_delta(
        self,
        *,
        parent_state: _ConcreteFakeYamlState,
        delta_ref: object,
        branch_from_parent: object | None = None,
    ) -> _ConcreteFakeYamlState:
        """Exporter tests do not restore deltas."""
        raise AssertionError(
            "Exporter tests should not load deltas: "
            f"{parent_state.node_id=} {branch_from_parent=!r} {delta_ref!r}"
        )

    def dump_state_summary(
        self, state: _ConcreteFakeYamlState
    ) -> _FakeCheckpointStateSummary:
        """Return lightweight state metadata for the payload."""
        self.dumped_summary_node_ids.append(state.node_id)
        return _FakeCheckpointStateSummary(tag=state.tag, node_id=state.node_id)


_CHILDREN_BY_ID = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [],
    4: [],
}

_DAG_CHILDREN_BY_ID = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: [],
}


class _SelectiveDeltaCheckpointCodec(_FakeIncrementalStateCheckpointCodec):
    """Fake codec that can reject deltas for selected parent-child pairs."""

    def __init__(self, invalid_pairs: set[tuple[int, int]]) -> None:
        """Store the parent-child pairs that should fail delta export."""
        super().__init__()
        self.invalid_pairs = invalid_pairs

    def dump_delta_from_parent(
        self,
        *,
        parent_state: _ConcreteFakeYamlState,
        child_state: _ConcreteFakeYamlState,
        branch_from_parent: object | None = None,
    ) -> object:
        """Reject configured parent-child pairs and delegate all others."""
        if (parent_state.node_id, child_state.node_id) in self.invalid_pairs:
            raise ValueError("delta rejected for this state parent")
        return super().dump_delta_from_parent(
            parent_state=parent_state,
            child_state=child_state,
            branch_from_parent=branch_from_parent,
        )


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
    children_by_id: dict[int, list[int]] = _CHILDREN_BY_ID,
    index_computation: IndexComputationType | None = None,
) -> Any:
    """Build one small real search runtime over the fake YAML domain."""
    starting_state = _ConcreteFakeYamlState(
        node_id=0,
        children_by_id=children_by_id,
        turn=Color.WHITE,
    )
    return create_search(
        state_type=_ConcreteFakeYamlState,
        dynamics=FakeYamlDynamics(),
        starting_state=starting_state,
        args=_build_args(index_computation=index_computation),
        random_generator=Random(0),
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(
            {node_id: float(node_id) for node_id in children_by_id}
        ),
        state_representation_factory=None,
    )


def _build_payload(
    runtime: Any,
    *,
    state_codec: _FakeIncrementalStateCheckpointCodec | None = None,
) -> SearchRuntimeCheckpointPayload:
    """Export a runtime using the fake incremental checkpoint codec."""
    return build_search_checkpoint_payload(
        runtime,
        state_codec=state_codec or _FakeIncrementalStateCheckpointCodec(),
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


def test_node_state_payloads_use_anchor_and_delta_codec_outputs() -> None:
    """Exported nodes should use explicit anchor/delta payloads from the codec."""
    runtime = _build_runtime()
    runtime.step()
    codec = _FakeIncrementalStateCheckpointCodec()

    payload = _build_payload(runtime, state_codec=codec)
    payload_nodes = _nodes_by_id(payload)
    root_payload = payload_nodes[runtime.tree.root_node.id]

    assert isinstance(root_payload.state_payload, AnchorCheckpointStatePayload)
    assert root_payload.state_payload.anchor_ref == {
        "kind": "anchor",
        "node_id": root_payload.node_id,
        "turn": runtime.tree.root_node.state.turn.name,
    }
    assert root_payload.state_payload.state_summary == _FakeCheckpointStateSummary(
        tag=runtime.tree.root_node.state.tag,
        node_id=runtime.tree.root_node.state.node_id,
    )

    non_root_payloads = [
        payload_nodes[node.id]
        for tree_depth in runtime.tree.descendants.range()
        for node in runtime.tree.descendants[tree_depth].values()
        if node.id != runtime.tree.root_node.id
    ]
    assert non_root_payloads
    assert all(
        isinstance(node_payload.state_payload, DeltaCheckpointStatePayload)
        for node_payload in non_root_payloads
    )
    assert codec.dumped_anchor_node_ids == [runtime.tree.root_node.state.node_id]
    assert all(
        node_payload.state_payload.delta_ref["branch_from_parent"]
        == node_payload.state_payload.state_parent_branch
        for node_payload in non_root_payloads
    )
    assert all(
        node_payload.state_payload.state_parent_node_id == node_payload.parent_node_id
        for node_payload in non_root_payloads
    )
    assert len(codec.dumped_delta_items) == len(non_root_payloads)
    assert sorted(codec.dumped_summary_node_ids) == sorted(payload_nodes)


def test_root_is_anchor_and_non_root_nodes_use_deltas_under_default_policy() -> None:
    """The default exporter policy should anchor the root and delta-encode children."""
    runtime = _build_runtime()
    runtime.step()

    payload_nodes = _nodes_by_id(_build_payload(runtime))
    root_payload = payload_nodes[runtime.tree.root_node.id]

    assert isinstance(root_payload.state_payload, AnchorCheckpointStatePayload)
    for branch_child in runtime.tree.root_node.branches_children.values():
        assert branch_child is not None
        child_payload = payload_nodes[branch_child.id]
        assert isinstance(child_payload.state_payload, DeltaCheckpointStatePayload)


def test_build_uses_codec_valid_state_parent_instead_of_graph_parent() -> None:
    """Delta export should try other DAG parents when the graph parent is invalid."""
    runtime = _build_runtime(children_by_id=_DAG_CHILDREN_BY_ID)
    runtime.step()
    runtime.step()
    shared_node = next(
        node
        for node in runtime.tree.descendants[2].values()
        if node.state.node_id == 3
    )
    representative_parent_id = next(iter(shared_node.parent_nodes)).id
    valid_parent_id = next(
        parent.id
        for parent in shared_node.parent_nodes
        if parent.id != representative_parent_id
    )
    codec = _SelectiveDeltaCheckpointCodec({(representative_parent_id, 3)})

    payload = _build_payload(runtime, state_codec=codec)
    shared_payload = next(
        node
        for node in payload.tree.nodes
        if node.depth == 2
        and isinstance(node.state_payload, DeltaCheckpointStatePayload)
        and cast("dict[str, object]", node.state_payload.delta_ref)["child_node_id"] == 3
    )

    assert shared_payload.parent_node_id == representative_parent_id
    assert shared_payload.state_payload.state_parent_node_id == valid_parent_id
    assert cast("dict[str, object]", shared_payload.state_payload.delta_ref)[
        "parent_node_id"
    ] == valid_parent_id


def test_build_falls_back_to_anchor_when_no_state_parent_can_emit_delta() -> None:
    """Delta export should anchor a DAG node when every candidate parent fails."""
    runtime = _build_runtime(children_by_id=_DAG_CHILDREN_BY_ID)
    runtime.step()
    runtime.step()
    codec = _SelectiveDeltaCheckpointCodec({(1, 3), (2, 3)})

    payload = _build_payload(runtime, state_codec=codec)
    shared_payload = next(
        node
        for node in payload.tree.nodes
        if node.depth == 2
        and node.state_payload.state_summary
        == _FakeCheckpointStateSummary(tag=3, node_id=3)
    )

    assert isinstance(shared_payload.state_payload, AnchorCheckpointStatePayload)


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
