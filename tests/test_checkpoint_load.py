"""Tests for restoring live search runtimes from checkpoint payloads."""

from __future__ import annotations

import valanga
from dataclasses import dataclass
from random import Random
from typing import Any, Self, cast

import pytest
from valanga import BranchKey, Color, State, StateModifications, StateTag

from anemone import SearchArgs, create_search
from anemone.checkpoints import (
    build_search_checkpoint_payload,
    load_search_from_checkpoint_payload,
)
from anemone.checkpoints.value_serialization import (
    serialize_checkpoint_atom,
    serialize_value,
)
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
    """Tiny external state-ref codec used by restore tests."""

    def __init__(self, children_by_id: dict[int, list[int]]) -> None:
        """Store the domain graph needed to rebuild fake states."""
        self._children_by_id = children_by_id

    def dump_state_ref(self, state: FakeYamlState) -> object:
        """Return an opaque state reference produced outside Anemone."""
        return {
            "codec": "fake-yaml",
            "node_id": state.node_id,
            "turn": state.turn.name,
        }

    def load_state_ref(self, state_ref: object) -> _ConcreteFakeYamlState:
        """Rebuild a fake domain state from one external state reference."""
        data = cast("dict[str, object]", state_ref)
        return _ConcreteFakeYamlState(
            node_id=cast("int", data["node_id"]),
            children_by_id=self._children_by_id,
            turn=Color[cast("str", data["turn"])],
        )


class _CountingStateValueEvaluator(MasterStateValueEvaluatorFromYaml):
    """Fake evaluator that records whether restore triggered evaluation."""

    def __init__(self, value_by_id: dict[int, float]) -> None:
        """Initialize the evaluator with a call counter."""
        super().__init__(value_by_id)
        self.evaluated_items_count = 0

    def evaluate_batch_items(self, items: Any) -> Any:
        """Count evaluated batch items before delegating to the fake evaluator."""
        self.evaluated_items_count += len(items)
        return super().evaluate_batch_items(items)


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

type _TupleBranch = tuple[int, int, int, int]

_TUPLE_BRANCH_CHILDREN_BY_ID = {
    0: [((1, 2, 3, 4), 1), ((5, 6, 7, 8), 2)],
    1: [],
    2: [],
}


class _TupleBranchKeyGenerator:
    """Simple branch-key generator that yields tuple branch labels in order."""

    sort_branch_keys: bool = False

    def __init__(self, keys: list[_TupleBranch]) -> None:
        """Store the tuple branch keys exposed by one test state."""
        self._keys = keys
        self._index = 0

    @property
    def all_generated_keys(self) -> list[_TupleBranch]:
        """Return the full ordered key list for checkpoint consumers."""
        return list(self._keys)

    def __iter__(self) -> Self:
        """Reset iteration state before yielding branch keys."""
        self._index = 0
        return self

    def __next__(self) -> _TupleBranch:
        """Yield the next tuple branch key."""
        if self._index >= len(self._keys):
            raise StopIteration
        key = self._keys[self._index]
        self._index += 1
        return key

    def more_than_one(self) -> bool:
        """Report whether this generator exposes multiple keys."""
        return len(self._keys) > 1

    def get_all(self) -> list[_TupleBranch]:
        """Return all tuple branch keys without advancing iteration state."""
        return list(self._keys)

    def copy_with_reset(self) -> _TupleBranchKeyGenerator:
        """Return an equivalent generator rewound to the first key."""
        return _TupleBranchKeyGenerator(list(self._keys))


@dataclass(slots=True)
class _TupleBranchState(State):
    """Small test state whose outgoing branches are tuple labels."""

    node_id: int
    children_by_id: dict[int, list[tuple[_TupleBranch, int]]]
    turn: Color = Color.WHITE

    @property
    def tag(self) -> StateTag:
        """Return the stable integer state tag used by the evaluator."""
        return int(self.node_id)

    @property
    def branch_keys(self) -> _TupleBranchKeyGenerator:
        """Expose tuple branch labels in their configured order."""
        return _TupleBranchKeyGenerator(
            [branch for branch, _ in self.children_by_id.get(self.node_id, [])]
        )

    def branch_name_from_key(self, key: BranchKey) -> str:
        """Return a readable tuple-branch edge label for diagnostics."""
        return f"{self.node_id}->{key!r}"

    def is_game_over(self) -> bool:
        """Report whether this state has any outgoing tuple branches left."""
        return len(self.children_by_id.get(self.node_id, [])) == 0

    def copy(self, stack: bool, deep_copy_legal_moves: bool = True) -> Self:
        """Return a shallow copy compatible with the Valanga state contract."""
        del stack, deep_copy_legal_moves
        return _TupleBranchState(
            node_id=self.node_id,
            children_by_id=self.children_by_id,
            turn=self.turn,
        )

    def step(self, branch_key: BranchKey) -> StateModifications | None:
        """Mutate in place by following one tuple-labeled edge."""
        self.node_id = self._child_id_from_branch(branch_key)
        self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE
        return None

    def _child_id_from_branch(self, branch_key: BranchKey) -> int:
        """Resolve a tuple branch label to its configured child node id."""
        if not isinstance(branch_key, tuple):
            raise TypeError(
                f"branch_key must be a tuple label, got {type(branch_key)}: {branch_key!r}"
            )

        for candidate_branch, child_id in self.children_by_id.get(self.node_id, []):
            if candidate_branch == branch_key:
                return child_id

        raise ValueError(
            f"Invalid tuple branch {branch_key!r} for node {self.node_id}."
        )


class _TupleBranchDynamics:
    """Minimal search dynamics for tuple-branch checkpoint tests."""

    __anemone_search_dynamics__ = True

    def legal_actions(
        self, state: _TupleBranchState
    ) -> _TupleBranchKeyGenerator:
        """Return the tuple branch keys currently available from this state."""
        return state.branch_keys

    def step(
        self, state: _TupleBranchState, action: BranchKey, *, depth: int
    ) -> valanga.Transition[_TupleBranchState]:
        """Return the next state reached by one tuple branch label."""
        del depth
        child_id = state._child_id_from_branch(action)
        next_turn = Color.BLACK if state.turn == Color.WHITE else Color.WHITE
        next_state = _TupleBranchState(
            node_id=child_id,
            children_by_id=state.children_by_id,
            turn=next_turn,
        )
        return valanga.Transition(
            next_state=next_state,
            modifications=None,
            is_over=next_state.is_game_over(),
            over_event=None,
            info={},
        )

    def action_name(self, state: _TupleBranchState, action: BranchKey) -> str:
        """Return a stable string form for one tuple branch label."""
        return state.branch_name_from_key(action)

    def action_from_name(self, state: _TupleBranchState, name: str) -> BranchKey:
        """Resolve a tuple branch label from its diagnostic string form."""
        del state
        for branch, _child_id in _TUPLE_BRANCH_CHILDREN_BY_ID[0]:
            if repr(branch) == name or f"0->{branch!r}" == name:
                return branch
        raise ValueError(f"Unknown tuple branch name {name!r}.")


class _TupleBranchStateCheckpointCodec:
    """External state-ref codec for the tuple-branch fake domain."""

    def dump_state_ref(self, state: _TupleBranchState) -> object:
        """Return an opaque state reference produced outside Anemone."""
        return {
            "codec": "tuple-branch",
            "node_id": state.node_id,
            "turn": state.turn.name,
        }

    def load_state_ref(self, state_ref: object) -> _TupleBranchState:
        """Rebuild a tuple-branch state from one external state reference."""
        data = cast("dict[str, object]", state_ref)
        return _TupleBranchState(
            node_id=cast("int", data["node_id"]),
            children_by_id=_TUPLE_BRANCH_CHILDREN_BY_ID,
            turn=Color[cast("str", data["turn"])],
        )


def _values_for(children_by_id: dict[int, list[int]]) -> dict[int, float]:
    """Return deterministic values for every fake graph node."""
    return {node_id: float(node_id) for node_id in children_by_id}


def _build_args(
    *,
    index_computation: IndexComputationType | None = None,
) -> SearchArgs:
    """Return a deterministic search configuration for restore tests."""
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
            _values_for(children_by_id)
        ),
        state_representation_factory=None,
    )


def _roundtrip_runtime(
    runtime: Any,
    *,
    children_by_id: dict[int, list[int]] = _CHILDREN_BY_ID,
    index_computation: IndexComputationType | None = None,
) -> Any:
    """Export and restore one runtime using the fake external state codec."""
    codec = _FakeStateCheckpointCodec(children_by_id)
    payload = build_search_checkpoint_payload(runtime, state_codec=codec)
    return load_search_from_checkpoint_payload(
        payload,
        state_codec=codec,
        dynamics=FakeYamlDynamics(),
        args=_build_args(index_computation=index_computation),
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(
            _values_for(children_by_id)
        ),
        random_generator=Random(0),
        state_representation_factory=None,
    )


def _build_tuple_branch_runtime() -> Any:
    """Build a small search runtime whose live branch labels are tuples."""
    starting_state = _TupleBranchState(
        node_id=0,
        children_by_id=_TUPLE_BRANCH_CHILDREN_BY_ID,
        turn=Color.WHITE,
    )
    return create_search(
        state_type=_TupleBranchState,
        dynamics=_TupleBranchDynamics(),
        starting_state=starting_state,
        args=_build_args(),
        random_generator=Random(0),
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(
            _values_for({node_id: [child_id for _, child_id in children] for node_id, children in _TUPLE_BRANCH_CHILDREN_BY_ID.items()})
        ),
        state_representation_factory=None,
    )


def _roundtrip_tuple_branch_runtime(runtime: Any) -> Any:
    """Export and restore one runtime whose tree edges use tuple labels."""
    codec = _TupleBranchStateCheckpointCodec()
    payload = build_search_checkpoint_payload(runtime, state_codec=codec)
    return load_search_from_checkpoint_payload(
        payload,
        state_codec=codec,
        dynamics=_TupleBranchDynamics(),
        args=_build_args(),
        state_type=_TupleBranchState,
        master_state_value_evaluator=MasterStateValueEvaluatorFromYaml(
            _values_for({node_id: [child_id for _, child_id in children] for node_id, children in _TUPLE_BRANCH_CHILDREN_BY_ID.items()})
        ),
        random_generator=Random(0),
        state_representation_factory=None,
    )


def _runtime_nodes_by_id(runtime: Any) -> dict[int, Any]:
    """Return live runtime nodes keyed by node id."""
    return {
        node.id: node
        for tree_depth in runtime.tree.descendants.range()
        for node in runtime.tree.descendants[tree_depth].values()
    }


def _child_signature(node: Any) -> dict[object, int]:
    """Return a comparable child mapping for one live node."""
    return {
        serialize_checkpoint_atom(branch): child.id
        for branch, child in node.branches_children.items()
        if child is not None
    }


def _parent_signature(node: Any) -> dict[int, set[object]]:
    """Return a comparable parent-edge mapping for one live node."""
    return {
        parent.id: {serialize_checkpoint_atom(branch) for branch in branches}
        for parent, branches in node.parent_nodes.items()
    }


def _latest_expansion_signature(
    runtime: Any,
) -> list[tuple[int, int | None, object, bool]]:
    """Return comparable selector-visible latest expansion records."""
    return [
        (
            expansion.child_node.id,
            expansion.parent_node.id if expansion.parent_node is not None else None,
            serialize_checkpoint_atom(expansion.branch_key),
            expansion.creation_child_node,
        )
        for expansion in runtime.latest_tree_expansions
    ]


def _serialized_value(value: Any) -> object:
    """Return a comparable serialized value payload."""
    return serialize_value(value) if value is not None else None


def test_checkpoint_restore_roundtrip_preserves_tree_identity() -> None:
    """Restored runtimes should keep root id, node count, and edge structure."""
    runtime = _build_runtime()
    runtime.step()
    runtime.step()

    restored = _roundtrip_runtime(runtime)

    original_nodes = _runtime_nodes_by_id(runtime)
    restored_nodes = _runtime_nodes_by_id(restored)
    assert restored.tree.root_node.id == runtime.tree.root_node.id
    assert restored.tree.nodes_count == runtime.tree.nodes_count
    assert set(restored_nodes) == set(original_nodes)
    for node_id, original_node in original_nodes.items():
        restored_node = restored_nodes[node_id]
        assert restored_node.tree_depth == original_node.tree_depth
        assert _child_signature(restored_node) == _child_signature(original_node)
        assert _parent_signature(restored_node) == _parent_signature(original_node)


def test_checkpoint_restore_preserves_node_values() -> None:
    """Direct and backed-up values should be restored without recomputation."""
    runtime = _build_runtime()
    runtime.step()
    runtime.step()

    restored = _roundtrip_runtime(runtime)

    for node_id, original_node in _runtime_nodes_by_id(runtime).items():
        restored_node = _runtime_nodes_by_id(restored)[node_id]
        assert _serialized_value(restored_node.tree_evaluation.direct_value) == (
            _serialized_value(original_node.tree_evaluation.direct_value)
        )
        assert _serialized_value(restored_node.tree_evaluation.backed_up_value) == (
            _serialized_value(original_node.tree_evaluation.backed_up_value)
        )
        assert restored_node.tree_evaluation.direct_evaluation_version == (
            original_node.tree_evaluation.direct_evaluation_version
        )


def test_checkpoint_restored_runtime_can_continue_search() -> None:
    """A restored runtime should be functional enough for the next step."""
    runtime = _build_runtime()
    runtime.step()
    restored = _roundtrip_runtime(runtime)
    node_count_before = restored.tree.nodes_count

    restored.step()

    assert restored.tree.nodes_count > node_count_before


def test_checkpoint_restored_runtime_supports_reevaluation() -> None:
    """A restored runtime should still support evaluator hot-swap refreshes."""
    runtime = _build_runtime()
    runtime.step()
    restored = _roundtrip_runtime(runtime)
    root_children = [
        child
        for child in restored.tree.root_node.branches_children.values()
        if child is not None
    ]
    new_values = {node_id: float(node_id + 100) for node_id in _CHILDREN_BY_ID}

    restored.set_evaluator(MasterStateValueEvaluatorFromYaml(new_values))
    report = restored.reevaluate_nodes(root_children)

    assert report.evaluator_version == 1
    assert report.requested_count == len(root_children)
    assert report.reevaluated_count == len(root_children)
    assert report.changed_count == len(root_children)
    assert restored.tree.root_node.tree_evaluation.backed_up_value is not None


def test_checkpoint_restore_does_not_evaluate_checkpointed_nodes() -> None:
    """Loading should restore checkpointed values instead of recomputing them."""
    runtime = _build_runtime()
    runtime.step()
    codec = _FakeStateCheckpointCodec(_CHILDREN_BY_ID)
    payload = build_search_checkpoint_payload(runtime, state_codec=codec)
    evaluator = _CountingStateValueEvaluator(_values_for(_CHILDREN_BY_ID))

    restored = load_search_from_checkpoint_payload(
        payload,
        state_codec=codec,
        dynamics=FakeYamlDynamics(),
        args=_build_args(),
        state_type=_ConcreteFakeYamlState,
        master_state_value_evaluator=evaluator,
        random_generator=Random(0),
        state_representation_factory=None,
    )

    assert evaluator.evaluated_items_count == 0
    restored.step()
    assert evaluator.evaluated_items_count > 0


def test_checkpoint_restore_preserves_evaluator_version() -> None:
    """Runtime and node-evaluator version counters should survive restore."""
    runtime = _build_runtime()
    runtime.step()
    runtime.evaluator_version = 7
    assert runtime.tree_manager.node_evaluator is not None
    runtime.tree_manager.node_evaluator.current_evaluator_version = 7

    restored = _roundtrip_runtime(runtime)

    assert restored.evaluator_version == 7
    assert restored.tree_manager.node_evaluator is not None
    assert restored.tree_manager.node_evaluator.current_evaluator_version == 7


def test_checkpoint_restore_preserves_dag_parent_edges() -> None:
    """Shared children should keep all incoming parent edges after restore."""
    runtime = _build_runtime(children_by_id=_DAG_CHILDREN_BY_ID)
    runtime.step()
    runtime.step()

    restored = _roundtrip_runtime(runtime, children_by_id=_DAG_CHILDREN_BY_ID)
    restored_nodes = _runtime_nodes_by_id(restored)
    shared_nodes = [
        node
        for node in restored_nodes.values()
        if node.state.node_id == 3 and node.tree_depth == 2
    ]

    assert len(shared_nodes) == 1
    shared_node = shared_nodes[0]
    assert {parent.state.node_id for parent in shared_node.parent_nodes} == {1, 2}
    assert {
        parent.state.node_id: {serialize_checkpoint_atom(branch) for branch in branches}
        for parent, branches in shared_node.parent_nodes.items()
    } == {1: {0}, 2: {0}}


def test_checkpoint_restore_preserves_tuple_branch_labels() -> None:
    """Tuple branch labels should survive checkpoint build and restore unchanged."""
    runtime = _build_tuple_branch_runtime()
    runtime.step()

    restored = _roundtrip_tuple_branch_runtime(runtime)

    original_root = runtime.tree.root_node
    restored_root = restored.tree.root_node
    assert sorted(
        [
            (serialize_checkpoint_atom(branch), child.id)
            for branch, child in restored_root.branches_children.items()
            if child is not None
        ],
        key=lambda item: (repr(item[0]), item[1]),
    ) == sorted(
        [
            (serialize_checkpoint_atom(branch), child.id)
            for branch, child in original_root.branches_children.items()
            if child is not None
        ],
        key=lambda item: (repr(item[0]), item[1]),
    )
    assert {
        branch
        for branch, child in restored_root.branches_children.items()
        if child is not None
    } == {(1, 2, 3, 4), (5, 6, 7, 8)}


def test_checkpoint_restore_preserves_latest_expansions() -> None:
    """Selector-visible latest expansion records should be restored by node id."""
    runtime = _build_runtime()
    runtime.step()

    restored = _roundtrip_runtime(runtime)

    assert _latest_expansion_signature(restored) == _latest_expansion_signature(runtime)


def test_checkpoint_restore_preserves_exploration_index_data() -> None:
    """Exploration-index data should be rebuilt with checkpointed field values."""
    runtime = _build_runtime(index_computation=IndexComputationType.RECUR_ZIPF)
    runtime.step()

    restored = _roundtrip_runtime(
        runtime,
        index_computation=IndexComputationType.RECUR_ZIPF,
    )

    for node_id, original_node in _runtime_nodes_by_id(runtime).items():
        restored_node = _runtime_nodes_by_id(restored)[node_id]
        assert restored_node.exploration_index_data is not None
        assert original_node.exploration_index_data is not None
        assert type(restored_node.exploration_index_data) is type(
            original_node.exploration_index_data
        )
        assert restored_node.exploration_index_data.index == pytest.approx(
            original_node.exploration_index_data.index
        )
        assert (
            restored_node.exploration_index_data.zipf_factored_proba
            == pytest.approx(original_node.exploration_index_data.zipf_factored_proba)
        )
