"""Tests for the debug snapshot and visualization helpers."""

# ruff: noqa: D101, D102, D103

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from anemone.debug import DebugEdgeView, DotRenderer, TreeSnapshotAdapter
from anemone.trees.tree_visualization import display, display_special


@dataclass
class FakeState:
    tag: str


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


@dataclass
class FakeMinmaxEvaluation:
    direct_value: FakeValue | None = None
    minmax_value: FakeValue | None = None
    best_branch_sequence: list[str] = field(default_factory=list)
    over_event_candidate: FakeOverEvent | None = None

    def get_over_event_candidate(self) -> FakeOverEvent | None:
        return self.over_event_candidate


@dataclass
class FakeIndexData:
    index: float | None = None
    max_depth_descendants: int | None = None


@dataclass(eq=False)
class FakeNode:
    id_: int
    tree_depth_: int
    state_: FakeState | None = None
    tree_evaluation: FakeEvaluation | FakeMinmaxEvaluation | None = None
    exploration_index_data: FakeIndexData | None = None
    parent_nodes_: dict[FakeNode, str] = field(default_factory=dict)
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
    def parent_nodes(self) -> dict[FakeNode, str]:
        return self.parent_nodes_

    @property
    def branches_children(self) -> dict[str, FakeNode | None]:
        return self.branches_children_


class FakeDynamics:
    __anemone_search_dynamics__ = True

    def action_name(self, state: FakeState | None, action: str) -> str:
        _ = state
        return f"move:{action}"


def test_tree_snapshot_adapter_captures_nodes_and_edges() -> None:
    root = FakeNode(id_=1, tree_depth_=0)
    child = FakeNode(id_=2, tree_depth_=1, parent_nodes_={root: "a"})
    root.branches_children_["a"] = child

    snapshot = TreeSnapshotAdapter().snapshot(root)

    assert snapshot.root_id == "1"
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}

    assert set(nodes_by_id) == {"1", "2"}
    assert nodes_by_id["2"].parent_ids == ("1",)
    assert nodes_by_id["2"].label == "id=2\ndepth=1"
    assert snapshot.edges == (DebugEdgeView(parent_id="1", child_id="2", label=None),)


def test_dot_renderer_outputs_nodes_and_edge() -> None:
    root = FakeNode(id_=1, tree_depth_=0)
    child = FakeNode(id_=2, tree_depth_=1, parent_nodes_={root: "a"})
    root.branches_children_["a"] = child

    snapshot = TreeSnapshotAdapter().snapshot(root)
    source = DotRenderer().render(snapshot).source

    assert '1 [label="id=1' in source
    assert '2 [label="id=2' in source
    assert "depth=0" in source
    assert "depth=1" in source
    assert "1 -> 2" in source


def test_tree_snapshot_adapter_supports_dag_parent_links() -> None:
    root_a = FakeNode(id_=1, tree_depth_=0)
    root_b = FakeNode(id_=2, tree_depth_=0)
    shared = FakeNode(id_=3, tree_depth_=1, parent_nodes_={root_a: "a", root_b: "b"})
    root_a.branches_children_["a"] = shared
    root_b.branches_children_["b"] = shared

    snapshot = TreeSnapshotAdapter().snapshot(root_a)
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}

    assert snapshot.root_id == "1"
    assert set(nodes_by_id["3"].parent_ids) == {"1", "2"}


def test_tree_snapshot_adapter_includes_state_tag_when_present() -> None:
    node = FakeNode(id_=4, tree_depth_=2, state_=FakeState(tag="mid"))

    snapshot = TreeSnapshotAdapter().snapshot(node)

    assert snapshot.nodes[0].label == "id=4\ndepth=2\nstate=mid"


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

    snapshot = TreeSnapshotAdapter().snapshot(node)

    assert snapshot.nodes[0].label == "\n".join(
        [
            "id=7",
            "depth=3",
            "state=state-tag",
            "direct=score=0.75, certainty=high",
            "backed_up=score=1.0, over=mate",
            "pv=a -> b",
            "over=terminal",
            "index=1.5",
            "max_depth_descendants=4",
        ]
    )


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

    snapshot = TreeSnapshotAdapter().snapshot(node)
    label = snapshot.nodes[0].label

    assert "direct=score=0.25" in label
    assert "backed_up=score=0.5, certainty=forced" in label
    assert "pv=c" in label
    assert "over=forced-win" in label


def test_tree_visualization_display_does_not_require_dot_description() -> None:
    root = FakeNode(id_=1, tree_depth_=0, state_=FakeState(tag="root"))
    child = FakeNode(
        id_=2,
        tree_depth_=1,
        state_=FakeState(tag="child"),
        parent_nodes_={root: "a"},
    )
    root.branches_children_["a"] = child

    assert not hasattr(root, "dot_description")

    source = display(
        tree=SimpleNamespace(root_node=root),
        format_str="svg",
        dynamics=FakeDynamics(),
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
        parent_nodes_={root: "a"},
    )
    root.branches_children_["a"] = child

    source = display_special(
        node=root,
        format_str="svg",
        index={"a": "rank-1"},
        dynamics=FakeDynamics(),
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
        parent_nodes_={root: "a"},
    )
    root.branches_children_["a"] = child

    source = display_special(
        node=root,
        format_str="svg",
        index={},
        dynamics=FakeDynamics(),
    ).source

    assert '1 -> 2 [label="?|move:a"]' in source
