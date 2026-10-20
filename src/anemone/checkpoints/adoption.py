"""Adopt committed state references on an existing, quiescent search graph.

This is an explicit caller-owned persistence boundary, never an automatic save
side effect. It reproduces the existing loader's handles and incoming-parent
order without constructing nodes or changing serialization policy.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from weakref import ReferenceType, ref

from anemone.utils.logger import checkpoint_logger

from .build import _maybe_dump_rng_state
from .build_atoms import _serialize_branch_collection
from .build_context import _CheckpointBuildContext, _CheckpointBuildMetrics
from .build_node_payloads import _serialize_linked_children
from .metadata import _validate_payload
from .node_restore import _checkpoint_summary_tag
from .payloads import DeltaCheckpointStatePayload
from .rng_state import rollout_random_generator
from .state_handles_restore import (
    _create_state_handles_from_node_ids,
    _create_state_resolver,
)

if TYPE_CHECKING:
    from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
    from anemone.tree_exploration import TreeExploration

    from ._protocols import IncrementalStateCheckpointCodec
    from .payloads import SearchRuntimeCheckpointPayload
    from .state_handles import CheckpointStateResolver


class CheckpointAdoptionError(ValueError):
    """Reject a stale or incompatible adoption before modifying any node."""

    def __init__(self, reason: str) -> None:
        """Describe the failed lifecycle precondition."""
        super().__init__(f"Cannot adopt checkpoint state: {reason}")


@dataclass(frozen=True, slots=True)
class CheckpointAdoptionStats:
    """Compact adoption measurements; no graph or state references."""

    nodes_bound: int
    parent_orders_changed: int


def _runtime_identity(search: TreeExploration[Any]) -> object:
    """Snapshot metadata without drawing or restoring either random stream."""
    rollout_rng = rollout_random_generator(search)
    return (
        search.evaluator_version,
        _maybe_dump_rng_state(search),
        None if rollout_rng is None else rollout_rng.getstate(),
    )


def _graph_identity(
    search: TreeExploration[Any],
) -> tuple[dict[int, AlgorithmNode[Any]], bytes]:
    """Index existing nodes and fingerprint structure, never reading node.state."""
    nodes: dict[int, AlgorithmNode[Any]] = {}
    digest = hashlib.sha256()
    context = _CheckpointBuildContext(metrics=_CheckpointBuildMetrics())
    outgoing_count = incoming_count = 0
    for depth, tag, node in search.tree.descendants.iter_on_all_nodes():
        if node.id in nodes:
            raise CheckpointAdoptionError("duplicate_live_node_ids")
        nodes[node.id] = node
        children = _serialize_linked_children(node.iter_child_links(), context=context)
        outgoing_count += len(children)
        for parent, branches in node.iter_parent_items():
            incoming_count += len(branches)
            if any(parent.child_for_branch(branch) is not node for branch in branches):
                raise CheckpointAdoptionError("inconsistent_incoming_edges")
        digest.update(
            repr(
                (
                    id(node),
                    node.id,
                    depth,
                    repr(tag),
                    node.tree_depth,
                    node.all_branches_generated,
                    _serialize_branch_collection(
                        node.iter_unopened_branches(), context=context
                    ),
                    children,
                )
            ).encode()
        )
    for node in nodes.values():
        if any(
            nodes.get(parent.id) is not parent for parent, _ in node.iter_parent_items()
        ):
            raise CheckpointAdoptionError("incoming_parent_not_in_live_graph")
        if any(nodes.get(child.id) is not child for child in node.iter_child_nodes()):
            raise CheckpointAdoptionError("outgoing_child_not_in_live_graph")
    if (
        len(nodes) != search.tree.nodes_count
        or nodes.get(search.tree.root_node.id) is not search.tree.root_node
        or incoming_count != outgoing_count
        or outgoing_count != search.tree.branch_count
    ):
        raise CheckpointAdoptionError("inconsistent_live_graph")
    return nodes, digest.digest()


def _validate_ancestry(payload: SearchRuntimeCheckpointPayload) -> None:
    """Check opaque delta ancestry in O(N), without replaying any state chain."""
    parents = {
        node.node_id: node.state_payload.state_parent_node_id
        for node in payload.tree.nodes
        if isinstance(node.state_payload, DeltaCheckpointStatePayload)
    }
    visited: set[int] = set()
    for node_id in parents:
        chain: set[int] = set()
        cursor = node_id
        while cursor in parents and cursor not in visited:
            if cursor in chain:
                raise CheckpointAdoptionError("cyclic_state_ancestry")
            chain.add(cursor)
            cursor = parents[cursor]
        visited.update(chain)


@dataclass(slots=True)
class CheckpointAdoption:
    """A compact, single-use plan prepared while the saved payload is available.

    The plan holds a weak runtime reference, scalar identities, node IDs and one
    lazy payload store/resolver. It does not retain nodes or node-runtime payloads.
    Finish checkpoint and generation publication before calling ``adopt``. Drop
    this plan on write/publication failure. Keep the runtime quiescent in between;
    no concurrent graph mutation is supported. Opaque state refs must stay valid
    and immutable, just as they must for ordinary checkpoint-backed handles.
    """

    _runtime_ref: ReferenceType[TreeExploration[Any]]
    _node_order: tuple[int, ...]
    _resolver: CheckpointStateResolver[Any]
    _graph_signature: bytes
    _metadata_identity: object
    _consumed: bool = False

    def adopt(self) -> CheckpointAdoptionStats:
        """Adopt after successful commit; preserve node/cache owners and values/RNG.

        All normal validation and handle allocation precede graph mutation. Old
        handles/resolvers remain valid for other owners and are never cleared.
        New handles share an initially empty resolver, as on a cold restore.
        """
        if self._consumed:
            raise CheckpointAdoptionError("plan_already_consumed")
        search = self._runtime_ref()
        if search is None:
            raise CheckpointAdoptionError("runtime_no_longer_alive")
        nodes, signature = _graph_identity(search)
        if (
            signature != self._graph_signature
            or _runtime_identity(search) != self._metadata_identity
        ):
            raise CheckpointAdoptionError("runtime_changed_since_prepare")
        handles = _create_state_handles_from_node_ids(
            node_ids=self._node_order,
            state_resolver=self._resolver,
        )
        rank = {node_id: index for index, node_id in enumerate(self._node_order)}
        orders = [
            (
                node,
                sorted(
                    (parent for parent, _ in node.iter_parent_items()),
                    key=lambda parent: rank[parent.id],
                ),
            )
            for node in nodes.values()
            if node.parent_count() > 1
        ]
        changed = 0
        for node, order in orders:
            if list(parent for parent, _ in node.iter_parent_items()) != order:
                node.tree_node.reorder_parent_links(order)
                changed += 1
        for node_id, handle in handles.items():
            nodes[node_id].tree_node.state_handle_ = handle
        self._consumed = True
        stats = CheckpointAdoptionStats(len(nodes), changed)
        checkpoint_logger.info(
            "[checkpoint-adoption] nodes_bound=%s parent_orders_changed=%s state_decodes=0",
            stats.nodes_bound,
            stats.parent_orders_changed,
        )
        return stats


def prepare_checkpoint_adoption(
    search: TreeExploration[Any],
    payload: SearchRuntimeCheckpointPayload,
    *,
    state_codec: IncrementalStateCheckpointCodec[Any],
) -> CheckpointAdoption:
    """Prepare adoption of this runtime's just-built checkpoint, without mutation.

    This checks structural identity, known summary tags and metadata, not arbitrary
    opaque codec contents. Supply the unmodified payload built from this runtime;
    it is not a loader for unrelated/untrusted checkpoints. No durability claim is
    inferred from payload construction. The caller must commit before ``adopt``.
    """
    _validate_payload(payload)
    _validate_ancestry(payload)
    nodes, signature = _graph_identity(search)
    if (
        set(nodes) != {node.node_id for node in payload.tree.nodes}
        or search.tree.root_node.id != payload.tree.root_node_id
    ):
        raise CheckpointAdoptionError("checkpoint_node_identity_mismatch")
    context = _CheckpointBuildContext(metrics=_CheckpointBuildMetrics())
    tags = {
        node.id: tag for _, tag, node in search.tree.descendants.iter_on_all_nodes()
    }
    for saved in payload.tree.nodes:
        node = nodes[saved.node_id]
        if (
            node.tree_depth != saved.depth
            or node.all_branches_generated != saved.generated_all_branches
            or _serialize_branch_collection(
                node.iter_unopened_branches(), context=context
            )
            != saved.unopened_branches
            or _serialize_linked_children(node.iter_child_links(), context=context)
            != saved.linked_children
        ):
            raise CheckpointAdoptionError("checkpoint_structure_mismatch")
        saved_tag = _checkpoint_summary_tag(saved.state_payload.state_summary)
        if saved_tag is not None and tags[saved.node_id] != saved_tag:
            raise CheckpointAdoptionError("checkpoint_state_tag_mismatch")
    # JSON round trips turn RNG tuples into lists; compare their encoded values.

    metadata = _runtime_identity(search)
    expected = (payload.evaluator_version, payload.rng_state, payload.rollout_rng_state)
    if json.dumps(metadata) != json.dumps(expected):
        raise CheckpointAdoptionError("checkpoint_runtime_metadata_mismatch")
    return CheckpointAdoption(
        _runtime_ref=ref(search),
        _node_order=tuple(node.node_id for node in payload.tree.nodes),
        _resolver=_create_state_resolver(payload=payload, state_codec=state_codec),
        _graph_signature=signature,
        _metadata_identity=metadata,
    )
