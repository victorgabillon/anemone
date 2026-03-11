from __future__ import annotations

from dataclasses import dataclass, field

from anemone.debug import DotRenderer, TreeSnapshotAdapter


@dataclass(eq=False)
class FakeNode:
    id_: int
    tree_depth_: int
    parent_nodes_: dict["FakeNode", str] = field(default_factory=dict)
    branches_children_: dict[str, "FakeNode | None"] = field(default_factory=dict)

    @property
    def id(self) -> int:
        return self.id_

    @property
    def tree_depth(self) -> int:
        return self.tree_depth_

    @property
    def parent_nodes(self) -> dict["FakeNode", str]:
        return self.parent_nodes_

    @property
    def branches_children(self) -> dict[str, "FakeNode | None"]:
        return self.branches_children_


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


def test_dot_renderer_outputs_nodes_and_edge() -> None:
    root = FakeNode(id_=1, tree_depth_=0)
    child = FakeNode(id_=2, tree_depth_=1, parent_nodes_={root: "a"})
    root.branches_children_["a"] = child

    snapshot = TreeSnapshotAdapter().snapshot(root)
    source = DotRenderer().render(snapshot).source

    assert '1 [label="id=1\ndepth=0"]' in source
    assert '2 [label="id=2\ndepth=1"]' in source
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
