"""Tests for the debug snapshot and visualization helpers."""

# pylint: disable=missing-class-docstring,missing-function-docstring,unused-import
# ruff: noqa: D101, D102, D103

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

from anemone.debug import (
    DebugEdgeView,
    DotRenderer,
    EvaluationDebugInspectorResolver,
    NodeDebugLabelBuilder,
    TreeSnapshotAdapter,
)
from anemone.debug.evaluation_inspectors import (
    GenericValueFamilyInspector,
    MinmaxFamilyInspector,
)
from anemone.trees.tree_visualization import display, display_special


@dataclass
class FakeState:
    tag: str
    turn: object | None = None


@dataclass
class FakeOverEvent:
    tag: str

    def get_over_tag(self) -> str:
        return self.tag


@dataclass
class FakeValue:
    score: float
    certainty: str | None = None
    over_event: FakeOverEvent | None = None


@dataclass
class FakeEvaluation:
    direct_value: FakeValue | None = None
    backed_up_value: FakeValue | None = None
    best_branch_sequence: list[str] = field(default_factory=list)
    over_event: FakeOverEvent | None = None
    exact: bool = False
    terminal: bool = False

    def has_exact_value(self) -> bool:
        return self.exact

    def is_terminal(self) -> bool:
        return self.terminal


@dataclass
class FakeMinmaxEvaluation:
    direct_value: FakeValue | None = None
    minmax_value: FakeValue | None = None
    best_branch_sequence: list[str] = field(default_factory=list)
    over_event_candidate: FakeOverEvent | None = None
    exact: bool = False
    terminal: bool = False

    def get_over_event_candidate(self) -> FakeOverEvent | None:
        return self.over_event_candidate

    def has_exact_value(self) -> bool:
        return self.exact

    def is_terminal(self) -> bool:
        return self.terminal


@dataclass
class FakeHybridEvaluation:
    direct_value: FakeValue | None = None
    backed_up_value: FakeValue | None = None
    minmax_value: FakeValue | None = None
    best_branch_sequence: list[str] = field(default_factory=list)
    over_event: FakeOverEvent | None = None
    exact: bool = False
    terminal: bool = False

    def has_exact_value(self) -> bool:
        return self.exact

    def is_terminal(self) -> bool:
        return self.terminal


@dataclass
class FakeIndexData:
    index: float | None = None
    max_depth_descendants: int | None = None


@dataclass(eq=False)
class FakeNode:
    id_: int
    tree_depth_: int
    state_: FakeState | None = None
    tree_evaluation: (
        FakeEvaluation | FakeMinmaxEvaluation | FakeHybridEvaluation | None
    ) = None
    exploration_index_data: FakeIndexData | None = None
    parent_nodes_: dict[FakeNode, set[str]] = field(default_factory=dict)
    branches_children_: dict[str, FakeNode | None] = field(default_factory=dict)

    @property
    def id(self) -> int:
        return self.id_

    @property
    def state(self) -> FakeState | None:
        return self.state_

    @property
    def tree_depth(self) -> int:
        return self.tree_depth_

    @property
    def parent_nodes(self) -> dict[FakeNode, set[str]]:
        return self.parent_nodes_

    @property
    def branches_children(self) -> dict[str, FakeNode | None]:
        return self.branches_children_


class FakeDynamics:
    __anemone_search_dynamics__ = True

    def action_name(self, state: FakeState | None, action: str) -> str:
        _ = state
        return f"move:{action}"


@dataclass(frozen=True)
class FakeTurn:
    name: str


def test_generic_value_family_inspector_summarizes_canonical_fields() -> None:
    evaluation = FakeEvaluation(
        direct_value=FakeValue(score=0.3, certainty="estimate"),
        backed_up_value=FakeValue(score=0.9, certainty="forced"),
        best_branch_sequence=["a", "b"],
        over_event=FakeOverEvent("win"),
    )

    lines = GenericValueFamilyInspector().summarize(evaluation)

    assert "direct=score=0.3, certainty=estimate" in lines
    assert "backed_up=score=0.9, certainty=forced" in lines
    assert "pv=a -> b" in lines
    assert "over=win" in lines


def test_minmax_family_inspector_summarizes_compatibility_fields() -> None:
    evaluation = FakeMinmaxEvaluation(
        direct_value=FakeValue(score=0.25),
        minmax_value=FakeValue(score=0.5, certainty="forced"),
        best_branch_sequence=["c"],
        over_event_candidate=FakeOverEvent("forced-win"),
    )

    lines = MinmaxFamilyInspector().summarize(evaluation)

    assert "direct=score=0.25" in lines
    assert "backed_up=score=0.5, certainty=forced" in lines
    assert "pv=c" in lines
    assert "over=forced-win" in lines


def test_evaluation_debug_inspector_resolver_prefers_generic_family() -> None:
    evaluation = FakeHybridEvaluation(
        direct_value=FakeValue(score=0.3),
        backed_up_value=FakeValue(score=0.7),
        minmax_value=FakeValue(score=1.1),
        best_branch_sequence=["a"],
        over_event=FakeOverEvent("resolved"),
    )

    lines = EvaluationDebugInspectorResolver().summarize(evaluation)

    assert "backed_up=score=0.7" in lines
    assert "backed_up=score=1.1" not in lines


def test_tree_snapshot_adapter_captures_nodes_and_edges() -> None:
    root = FakeNode(id_=1, tree_depth_=0)
    child = FakeNode(id_=2, tree_depth_=1, parent_nodes_={root: {"a"}})
    root.branches_children_["a"] = child

    snapshot = TreeSnapshotAdapter().snapshot(cast("Any", root))

    assert snapshot.root_id == "1"
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}

    assert set(nodes_by_id) == {"1", "2"}
    assert nodes_by_id["2"].parent_ids == ("1",)
    assert nodes_by_id["2"].label == "id=2\ndepth=1"
    assert nodes_by_id["1"].child_ids == ("2",)
    assert nodes_by_id["1"].edge_labels_by_child == {}
    assert snapshot.edges == (DebugEdgeView(parent_id="1", child_id="2", label=None),)


def test_dot_renderer_outputs_nodes_and_edge() -> None:
    root = FakeNode(id_=1, tree_depth_=0)
    child = FakeNode(id_=2, tree_depth_=1, parent_nodes_={root: {"a"}})
    root.branches_children_["a"] = child

    snapshot = TreeSnapshotAdapter().snapshot(cast("Any", root))
    source = DotRenderer().render(snapshot).source

    assert '1 [label="id=1' in source
    assert '2 [label="id=2' in source
    assert "depth=0" in source
    assert "depth=1" in source
    assert "1 -> 2" in source


def test_dot_renderer_styles_player_and_terminal_nodes() -> None:
    root = FakeNode(
        id_=1,
        tree_depth_=0,
        state_=FakeState(tag="root", turn=FakeTurn(name="WHITE")),
    )
    child = FakeNode(
        id_=2,
        tree_depth_=1,
        state_=FakeState(tag="reply", turn=FakeTurn(name="BLACK")),
        parent_nodes_={root: {"a"}},
        tree_evaluation=FakeEvaluation(
            over_event=FakeOverEvent("done"),
            exact=True,
            terminal=True,
        ),
    )
    root.branches_children_["a"] = child

    snapshot = TreeSnapshotAdapter().snapshot(cast("Any", root))
    source = DotRenderer().render(snapshot).source

    assert "player=MAX" in source
    assert "player=MIN" in source
    assert "color=WHITE" in source
    assert "color=BLACK" in source
    assert '1 [label="id=1\ndepth=0\nplayer=MAX\ncolor=WHITE\nstate=root"' in source
    assert "shape=box" in source
    assert 'fillcolor="#d8e7ff"' in source
    assert 'color="#456fb3"' in source
    assert (
        '2 [label="id=2\ndepth=1\nplayer=MIN\ncolor=BLACK\nstate=reply\nover=done"'
        in source
    )
    assert 'fillcolor="#dcefd9"' in source
    assert 'color="#2f6b2f"' in source


def test_dot_renderer_styles_forced_nodes_distinct_from_terminal_nodes() -> None:
    node = FakeNode(
        id_=3,
        tree_depth_=0,
        state_=FakeState(tag="forced", turn=FakeTurn(name="WHITE")),
        tree_evaluation=FakeEvaluation(
            over_event=FakeOverEvent("forced-win"),
            exact=True,
            terminal=False,
        ),
    )

    snapshot = TreeSnapshotAdapter().snapshot(cast("Any", node))
    source = DotRenderer().render(snapshot).source

    assert snapshot.nodes[0].is_exact is True
    assert snapshot.nodes[0].is_terminal is False
    assert 'fillcolor="#f3e5b1"' in source
    assert 'color="#8c6a12"' in source
    assert 'fillcolor="#dcefd9"' not in source


def test_tree_snapshot_adapter_supports_dag_parent_links() -> None:
    root_a = FakeNode(id_=1, tree_depth_=0)
    root_b = FakeNode(id_=2, tree_depth_=0)
    shared = FakeNode(
        id_=3,
        tree_depth_=1,
        parent_nodes_={root_a: {"a"}, root_b: {"b"}},
    )
    root_a.branches_children_["a"] = shared
    root_b.branches_children_["b"] = shared

    snapshot = TreeSnapshotAdapter().snapshot(cast("Any", root_a))
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}

    assert snapshot.root_id == "1"
    assert set(nodes_by_id["3"].parent_ids) == {"1", "2"}


def test_tree_snapshot_adapter_includes_state_tag_when_present() -> None:
    node = FakeNode(id_=4, tree_depth_=2, state_=FakeState(tag="mid"))

    snapshot = TreeSnapshotAdapter().snapshot(cast("Any", node))

    assert snapshot.nodes[0].label == "id=4\ndepth=2\nstate=mid"
    assert snapshot.nodes[0].state_tag == "mid"


def test_tree_snapshot_adapter_includes_player_label_when_turn_present() -> None:
    node = FakeNode(
        id_=5,
        tree_depth_=1,
        state_=FakeState(tag="mid", turn=FakeTurn(name="BLACK")),
    )

    snapshot = TreeSnapshotAdapter().snapshot(cast("Any", node))

    assert snapshot.nodes[0].label == (
        "id=5\ndepth=1\nplayer=MIN\ncolor=BLACK\nstate=mid"
    )
    assert snapshot.nodes[0].player_label == "MIN"


def test_tree_snapshot_adapter_includes_state_evaluation_and_index_lines() -> None:
    value = FakeValue(score=0.75, certainty="high")
    backed_up = FakeValue(score=1.0, over_event=FakeOverEvent("mate"))
    evaluation = FakeEvaluation(
        direct_value=value,
        backed_up_value=backed_up,
        best_branch_sequence=["a", "b"],
        over_event=FakeOverEvent("terminal"),
    )
    node = FakeNode(
        id_=7,
        tree_depth_=3,
        state_=FakeState(tag="state-tag"),
        tree_evaluation=evaluation,
        exploration_index_data=FakeIndexData(index=1.5, max_depth_descendants=4),
    )

    snapshot = TreeSnapshotAdapter().snapshot(cast("Any", node))

    assert snapshot.nodes[0].label == "\n".join(
        [
            "id=7",
            "depth=3",
            "state=state-tag",
            "direct=score=0.8, certainty=high",
            "backed_up=score=1.0, over=mate",
            "pv=a -> b",
            "over=terminal",
            "index=1.5",
            "max_depth_descendants=4",
        ]
    )
    assert snapshot.nodes[0].state_tag == "state-tag"
    assert snapshot.nodes[0].direct_value == "score=0.75, certainty=high"
    assert snapshot.nodes[0].backed_up_value == "score=1.0, over=mate"
    assert snapshot.nodes[0].principal_variation == "a -> b"
    assert snapshot.nodes[0].over_event == "terminal"
    assert snapshot.nodes[0].index_fields == {
        "index": "1.5",
        "max_depth_descendants": "4",
    }


def test_tree_snapshot_adapter_populates_child_ids_and_edge_labels() -> None:
    root = FakeNode(id_=10, tree_depth_=0)
    child_a = FakeNode(id_=11, tree_depth_=1, parent_nodes_={root: {"a"}})
    child_b = FakeNode(id_=12, tree_depth_=1, parent_nodes_={root: {"b"}})
    root.branches_children_["a"] = child_a
    root.branches_children_["b"] = child_b

    snapshot = TreeSnapshotAdapter(
        edge_label_builder=lambda parent, branch_key, child: f"{branch_key}:{child.id}"
    ).snapshot(cast("Any", root))
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}

    assert nodes_by_id["10"].child_ids == ("11", "12")
    assert nodes_by_id["10"].edge_labels_by_child == {
        "11": "a:11",
        "12": "b:12",
    }


def test_node_debug_label_builder_delegates_evaluation_and_index_formatting() -> None:
    node = FakeNode(
        id_=9,
        tree_depth_=4,
        state_=FakeState(tag="builder"),
        tree_evaluation=FakeEvaluation(
            direct_value=FakeValue(score=0.1),
            backed_up_value=FakeValue(score=0.8),
            best_branch_sequence=["x", "y"],
            over_event=FakeOverEvent("done"),
        ),
        exploration_index_data=FakeIndexData(index=2.5, max_depth_descendants=6),
    )

    label = NodeDebugLabelBuilder().build_label(node)

    assert "id=9" in label
    assert "depth=4" in label
    assert "state=builder" in label
    assert "direct=score=0.1" in label
    assert "backed_up=score=0.8" in label
    assert "pv=x -> y" in label
    assert "over=done" in label
    assert "index=2.5" in label
    assert "max_depth_descendants=6" in label


def test_tree_snapshot_adapter_uses_minmax_and_over_event_fallbacks() -> None:
    node = FakeNode(
        id_=8,
        tree_depth_=1,
        state_=FakeState(tag="fallback"),
        tree_evaluation=FakeMinmaxEvaluation(
            direct_value=FakeValue(score=0.25),
            minmax_value=FakeValue(score=0.5, certainty="forced"),
            best_branch_sequence=["c"],
            over_event_candidate=FakeOverEvent("forced-win"),
        ),
    )

    assert not hasattr(node, "dot_description")
    assert not hasattr(node.tree_evaluation, "dot_description")

    snapshot = TreeSnapshotAdapter().snapshot(cast("Any", node))
    label = snapshot.nodes[0].label

    assert "direct=score=0.2" in label
    assert "backed_up=score=0.5, certainty=forced" in label
    assert "pv=c" in label
    assert "over=forced-win" in label


def test_tree_visualization_display_does_not_require_dot_description() -> None:
    root = FakeNode(id_=1, tree_depth_=0, state_=FakeState(tag="root"))
    child = FakeNode(
        id_=2,
        tree_depth_=1,
        state_=FakeState(tag="child"),
        parent_nodes_={root: {"a"}},
    )
    root.branches_children_["a"] = child

    assert not hasattr(root, "dot_description")

    source = display(
        tree=cast("Any", SimpleNamespace(root_node=root)),
        format_str="svg",
        dynamics=cast("Any", FakeDynamics()),
    ).source

    assert '1 [label="id=1' in source
    assert '2 [label="id=2' in source
    assert "depth=0" in source
    assert "depth=1" in source
    assert "state=root" in source
    assert "state=child" in source
    assert '1 -> 2 [label="move:a"]' in source


def test_tree_visualization_display_special_preserves_legacy_edge_label_prefix() -> (
    None
):
    root = FakeNode(id_=1, tree_depth_=0, state_=FakeState(tag="root"))
    child = FakeNode(
        id_=2,
        tree_depth_=1,
        state_=FakeState(tag="child"),
        parent_nodes_={root: {"a"}},
    )
    root.branches_children_["a"] = child

    source = display_special(
        node=cast("Any", root),
        format_str="svg",
        index={"a": "rank-1"},
        dynamics=cast("Any", FakeDynamics()),
    ).source

    assert '1 [label="id=1' in source
    assert '2 [label="id=2' in source
    assert "state=root" in source
    assert "state=child" in source
    assert '1 -> 2 [label="rank-1|move:a"]' in source


def test_tree_visualization_display_special_uses_placeholder_for_missing_rank() -> None:
    root = FakeNode(id_=1, tree_depth_=0, state_=FakeState(tag="root"))
    child = FakeNode(
        id_=2,
        tree_depth_=1,
        state_=FakeState(tag="child"),
        parent_nodes_={root: {"a"}},
    )
    root.branches_children_["a"] = child

    source = display_special(
        node=cast("Any", root),
        format_str="svg",
        index={},
        dynamics=cast("Any", FakeDynamics()),
    ).source

    assert '1 -> 2 [label="?|move:a"]' in source
