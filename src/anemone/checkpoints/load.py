"""Restore live Anemone search runtimes from checkpoint payloads."""

# ruff: noqa: TC001

from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnusedFunction=false
from random import Random
from typing import TYPE_CHECKING

from anemone._runtime_assembly import assemble_search_runtime_dependencies
from anemone._valanga_types import AnyTurnState
from anemone.node_evaluation.tree.factory import NodeTreeMinmaxEvaluationFactory
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.progress_monitor.progress_monitor import create_stopping_criterion
from anemone.tree_exploration import TreeExploration
from anemone.tree_exploration_debug import (
    log_final_best_line,
    maybe_log_iteration_progress,
)

from .metadata import (
    CheckpointRestoreError,
    RestoreMemoryPhaseLogger,
    _checkpoint_state_payload_counts,
    _latest_expansion_count,
    _log_restore_memory_phase,
    _log_restore_phase,
    _restore_runtime_metadata,
    _restore_runtime_state,
    _validate_checkpoint_node_metadata,
    _validate_checkpoint_node_payloads,
    _validate_payload,
)
from .node_restore import (
    _build_tree,
    _build_tree_from_node_payloads,
    _build_tree_from_node_shells,
    _checkpoint_node_shell_state_tag,
    _checkpoint_node_state_tag,
    _checkpoint_summary_tag,
    _CheckpointNodeRuntime,
    _CheckpointNodeShell,
    _create_nodes,
    _create_nodes_from_node_payloads,
    _create_nodes_from_node_shells,
    _group_node_payloads_by_depth,
    _group_node_shells_by_depth,
    _link_nodes,
    _populate_descendants_for_restore,
    _populate_descendants_from_node_shells_for_restore,
    _restore_node_runtime_state,
)
from .payloads import SearchRuntimeCheckpointPayload
from .restore_atoms import (
    _deserialize_branch,
    _deserialize_optional_branch,
    _require_node,
)
from .selector_payloads import (
    _infer_latest_expansion_depth,
    _iter_selector_components,
    _restore_explicit_selector_state,
    _restore_inferred_depth_selector_state,
)
from .state_handles_restore import (
    _build_checkpoint_payload_store,
    _create_state_handles,
    _create_state_handles_from_node_ids,
    _create_state_handles_from_node_payloads,
    _create_state_resolver,
    _create_state_resolver_from_node_payloads,
    _create_state_resolver_from_payload_store,
    _node_ids_are_dense_zero_based,
    _node_payload_ids_are_dense_zero_based,
)
from .tree_expansions_payloads import (
    _restore_tree_expansion,
    _restore_tree_expansions,
    _restore_tree_expansions_runtime_state,
)

if TYPE_CHECKING:
    from valanga import Dynamics, RepresentationFactory, StateModifications
    from valanga.evaluator_types import EvaluatorInput
    from valanga.policy import NotifyProgressCallable

    from anemone.dynamics import SearchDynamics
    from anemone.factory import SearchArgs
    from anemone.hooks.search_hooks import SearchHooks
    from anemone.node_evaluation.direct.protocols import MasterStateValueEvaluator
    from anemone.node_evaluation.tree.factory import NodeTreeEvaluationFactory

    from ._protocols import IncrementalStateCheckpointCodec


def load_search_from_checkpoint_payload[
    StateT: AnyTurnState,
    ActionT,
](
    payload: SearchRuntimeCheckpointPayload,
    *,
    state_codec: IncrementalStateCheckpointCodec[StateT],
    dynamics: SearchDynamics[StateT, ActionT] | Dynamics[StateT],
    args: SearchArgs,
    state_type: type[StateT],
    master_state_value_evaluator: MasterStateValueEvaluator,
    random_generator: Random | None = None,
    state_representation_factory: RepresentationFactory[
        StateT, EvaluatorInput, StateModifications
    ]
    | None = None,
    node_tree_evaluation_factory: NodeTreeEvaluationFactory[StateT] | None = None,
    notify_progress: NotifyProgressCallable | None = None,
    hooks: SearchHooks | None = None,
    restore_memory_phase_logger: RestoreMemoryPhaseLogger | None = None,
) -> TreeExploration[AlgorithmNode[StateT]]:
    """Restore a live search runtime from one in-memory checkpoint payload.

    The loader restores exported tree, node, value, and evaluation-cache state
    directly without evaluating checkpointed nodes, and without eagerly
    materializing checkpointed domain states. Selector-visible latest
    expansions are restored exactly from the payload. Selector-private state is
    only restored where this module has an explicit, narrow runtime contract;
    currently that means inferring Uniform's depth cursor from the latest
    expansion depth. The supplied evaluator is kept for future expansions and
    explicit reevaluations after the runtime resumes.
    """
    anchor_count, delta_count = _checkpoint_state_payload_counts(payload)
    common_metadata = {
        "node_count": len(payload.tree.nodes),
        "root_node_id": payload.tree.root_node_id,
        "format_version": payload.format_version,
        "anchor_count": anchor_count,
        "delta_count": delta_count,
        "latest_expansion_count": _latest_expansion_count(payload),
    }
    _log_restore_memory_phase(
        restore_memory_phase_logger,
        "after_runtime_restore_start",
        **common_metadata,
        raw_checkpoint_referenced=False,
        typed_checkpoint_referenced=True,
    )
    with _log_restore_phase("validate_payload", **common_metadata):
        _validate_payload(payload)

    checkpoint_random_generator = random_generator if random_generator else Random()
    with _log_restore_phase("assemble_runtime_dependencies", **common_metadata):
        dependencies = assemble_search_runtime_dependencies(
            dynamics=dynamics,
            args=args,
            random_generator=checkpoint_random_generator,
            master_state_evaluator=master_state_value_evaluator,
            state_representation_factory=state_representation_factory,
            node_tree_evaluation_factory=(
                node_tree_evaluation_factory or NodeTreeMinmaxEvaluationFactory()
            ),
            hooks=hooks,
        )
    with _log_restore_phase("create_state_resolver", **common_metadata):
        state_resolver = _create_state_resolver(
            payload=payload,
            state_codec=state_codec,
        )
    _log_restore_memory_phase(
        restore_memory_phase_logger,
        "after_checkpoint_state_resolver_created",
        **common_metadata,
        raw_checkpoint_referenced=False,
        typed_checkpoint_referenced=True,
    )
    with _log_restore_phase("create_state_handles", **common_metadata):
        state_handles_by_id = _create_state_handles(
            payload=payload,
            state_resolver=state_resolver,
        )
    _ = state_type

    with _log_restore_phase("create_nodes", **common_metadata):
        nodes_by_id = _create_nodes(
            node_factory=dependencies.tree_manager.tree_manager.node_factory,
            payload=payload,
            state_handles_by_id=state_handles_by_id,
        )
    _log_restore_memory_phase(
        restore_memory_phase_logger,
        "after_runtime_tree_nodes_created",
        **common_metadata,
        raw_checkpoint_referenced=False,
        typed_checkpoint_referenced=True,
    )
    with _log_restore_phase("link_nodes", **common_metadata):
        _link_nodes(payload.tree.nodes, nodes_by_id=nodes_by_id)
    with _log_restore_phase("restore_node_runtime_state", **common_metadata):
        _restore_node_runtime_state(payload.tree.nodes, nodes_by_id=nodes_by_id)

    with _log_restore_phase("build_tree", **common_metadata):
        restored_tree = _build_tree(payload=payload, nodes_by_id=nodes_by_id)
    node_selector = dependencies.node_selector_create()
    with _log_restore_phase("create_runtime", **common_metadata):
        runtime = TreeExploration(
            tree=restored_tree,
            tree_manager=dependencies.tree_manager,
            stopping_criterion=create_stopping_criterion(
                args=args.stopping_criterion,
                node_selector=node_selector,
            ),
            node_selector=node_selector,
            recommend_branch_after_exploration=args.recommender_rule,
            notify_percent_function=notify_progress,
            iteration_progress_reporter=maybe_log_iteration_progress,
            search_result_reporter=log_final_best_line,
        )
    with _log_restore_phase("restore_runtime_state", **common_metadata):
        _restore_runtime_state(
            runtime=runtime,
            payload=payload,
            random_generator=checkpoint_random_generator,
        )
    with _log_restore_phase("restore_tree_expansions", **common_metadata):
        _restore_tree_expansions_runtime_state(
            runtime=runtime,
            payload=payload.latest_tree_expansions,
            nodes_by_id=nodes_by_id,
        )
    with _log_restore_phase("restore_explicit_selector_state", **common_metadata):
        _restore_explicit_selector_state(
            runtime=runtime,
            payload=payload.selector_state,
        )
    with _log_restore_phase("restore_selector_state", **common_metadata):
        _restore_inferred_depth_selector_state(
            runtime=runtime,
            payload=payload.latest_tree_expansions,
            nodes_by_id=nodes_by_id,
        )
    _log_restore_memory_phase(
        restore_memory_phase_logger,
        "after_runtime_restore_done",
        **common_metadata,
        branch_count=runtime.tree.branch_count,
        raw_checkpoint_referenced=False,
        typed_checkpoint_referenced=True,
    )
    return runtime


__all__ = [
    "CheckpointRestoreError",
    "RestoreMemoryPhaseLogger",
    "_CheckpointNodeRuntime",
    "_CheckpointNodeShell",
    "_build_checkpoint_payload_store",
    "_build_tree",
    "_build_tree_from_node_payloads",
    "_build_tree_from_node_shells",
    "_checkpoint_node_shell_state_tag",
    "_checkpoint_node_state_tag",
    "_checkpoint_state_payload_counts",
    "_checkpoint_summary_tag",
    "_create_nodes",
    "_create_nodes_from_node_payloads",
    "_create_nodes_from_node_shells",
    "_create_state_handles",
    "_create_state_handles_from_node_ids",
    "_create_state_handles_from_node_payloads",
    "_create_state_resolver",
    "_create_state_resolver_from_node_payloads",
    "_create_state_resolver_from_payload_store",
    "_deserialize_branch",
    "_deserialize_optional_branch",
    "_group_node_payloads_by_depth",
    "_group_node_shells_by_depth",
    "_infer_latest_expansion_depth",
    "_iter_selector_components",
    "_latest_expansion_count",
    "_link_nodes",
    "_log_restore_memory_phase",
    "_log_restore_phase",
    "_node_ids_are_dense_zero_based",
    "_node_payload_ids_are_dense_zero_based",
    "_populate_descendants_for_restore",
    "_populate_descendants_from_node_shells_for_restore",
    "_require_node",
    "_restore_explicit_selector_state",
    "_restore_inferred_depth_selector_state",
    "_restore_node_runtime_state",
    "_restore_runtime_metadata",
    "_restore_runtime_state",
    "_restore_tree_expansion",
    "_restore_tree_expansions",
    "_restore_tree_expansions_runtime_state",
    "_validate_checkpoint_node_metadata",
    "_validate_checkpoint_node_payloads",
    "_validate_payload",
    "load_search_from_checkpoint_payload",
]
