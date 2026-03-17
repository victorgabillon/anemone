"""Provide the single-agent node-evaluation protocol."""

from typing import Protocol

from valanga import State

from anemone.node_evaluation.tree.node_tree_evaluation import NodeTreeEvaluation


class NodeSingleAgentEvaluation[StateT: State = State](
    NodeTreeEvaluation[StateT], Protocol
):
    """Single-agent decision and backup semantics layered on canonical values.

    The generic tree-search surface is inherited from ``NodeTreeEvaluation``.
    """
