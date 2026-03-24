"""Provide compatibility wrappers for tree visualization."""


# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import sys
from pickle import dump
from typing import TYPE_CHECKING, Any

from valanga import State

from anemone.debug import DotRenderer, TreeSnapshotAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

    from graphviz import Digraph
    from valanga import BranchKey

    from anemone.dynamics import SearchDynamics
    from anemone.nodes import ITreeNode

    from .tree import Tree


def _action_edge_label_builder[StateT: State](
    dynamics: SearchDynamics[StateT, Any],
) -> Callable[[ITreeNode[StateT], BranchKey, ITreeNode[StateT]], str]:
    """Build edge labels from action names."""

    def build_label(
        parent: ITreeNode[StateT], branch_key: BranchKey, child: ITreeNode[StateT]
    ) -> str:
        _ = child
        return str(dynamics.action_name(parent.state, branch_key))

    return build_label


def _special_edge_label_builder[StateT: State](
    dynamics: SearchDynamics[StateT, Any],
    index: dict[BranchKey, str],
) -> Callable[[ITreeNode[StateT], BranchKey, ITreeNode[StateT]], str]:
    """Build the legacy special-edge label prefix with best-effort rank data."""

    def build_label(
        parent: ITreeNode[StateT], branch_key: BranchKey, child: ITreeNode[StateT]
    ) -> str:
        _ = child
        parts = [
            index.get(branch_key, "?"),
            str(dynamics.action_name(parent.state, branch_key)),
        ]
        return "|".join(parts)

    return build_label


def _render_snapshot[StateT: State](
    root: ITreeNode[StateT],
    format_str: str,
    edge_label_builder: (
        Callable[[ITreeNode[StateT], BranchKey, ITreeNode[StateT]], str | None] | None
    ),
) -> Digraph:
    """Capture ``root`` into a debug snapshot and render it as DOT."""
    snapshot = TreeSnapshotAdapter(edge_label_builder=edge_label_builder).snapshot(root)
    return DotRenderer().render(snapshot, format_str=format_str)


def add_dot[StateT: State](
    dot: Digraph, treenode: ITreeNode[StateT], dynamics: SearchDynamics[StateT, Any]
) -> None:
    """Add the subtree rooted at ``treenode`` to ``dot``."""
    rendered = _render_snapshot(
        root=treenode,
        format_str=getattr(dot, "format", None) or "pdf",
        edge_label_builder=_action_edge_label_builder(dynamics),
    )
    dot.body.extend(rendered.body)


def display_special(
    node: ITreeNode[Any],
    format_str: str,
    index: dict[BranchKey, str],
    dynamics: SearchDynamics[Any, Any],
) -> Digraph:
    """Display a tree with the legacy special-edge label prefix for the given node."""
    return _render_snapshot(
        root=node,
        format_str=format_str,
        edge_label_builder=_special_edge_label_builder(dynamics, index),
    )


def display[NodeT: ITreeNode[Any]](
    tree: Tree[NodeT],
    format_str: str,
    dynamics: SearchDynamics[Any, Any],
) -> Digraph:
    """Display a tree using the debug snapshot and DOT renderer."""
    return _render_snapshot(
        root=tree.root_node,
        format_str=format_str,
        edge_label_builder=_action_edge_label_builder(dynamics),
    )


def save_pdf_to_file[NodeT: ITreeNode[Any]](
    tree: Tree[NodeT], dynamics: SearchDynamics[Any, Any]
) -> None:
    """Save the visualization of a tree as a PDF file."""
    dot = display(tree=tree, format_str="pdf", dynamics=dynamics)
    tag_ = tree.root_node.tag
    dot.render("chipiron/runs/treedisplays/TreeVisual_" + str(tag_) + ".pdf")


def save_raw_data_to_file[NodeT: ITreeNode[Any]](
    tree: Tree[NodeT], count: str = "#"
) -> None:
    """Save raw tree data to a file."""
    tag_ = tree.root_node.tag
    filename = "chipiron/debugTreeData_" + str(tag_) + "-" + str(count) + ".td"

    sys.setrecursionlimit(100000)
    with open(filename, "wb") as f:
        dump([tree.descendants, tree.root_node], f)
