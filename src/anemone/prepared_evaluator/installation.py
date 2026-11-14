"""Preallocate owner-bound derived objects, then perform one checked installation."""

# Exact-type admission rejects subclasses; internal bindings deliberately reuse
# native checkpoint/runtime ownership. Exceptions below are limited to those sites.

# Runtime annotations match the existing native checkpoint helpers.
import json
from dataclasses import dataclass
from types import MethodType
from typing import Any, cast
from weakref import ref

from anemone.backup_policies.aggregation import BestChildAggregationPolicy
from anemone.backup_policies.explicit_max import ExplicitMaxBackupPolicy
from anemone.backup_policies.proof import MaxProofPolicy
from anemone.checkpoints.evaluation_payloads import (
    _restore_evaluation,  # pyright: ignore[reportPrivateUsage]
    _restore_exploration_index,  # pyright: ignore[reportPrivateUsage]
)
from anemone.checkpoints.rng_state import rollout_random_generator
from anemone.checkpoints.sharded_decode import (
    _exploration_index_payload_from_mapping,  # pyright: ignore[reportPrivateUsage]
    _node_evaluation_payload_from_mapping,  # pyright: ignore[reportPrivateUsage]
    _selector_payload_from_mapping,  # pyright: ignore[reportPrivateUsage]
)
from anemone.checkpoints.state_handles import CheckpointBackedStateHandle
from anemone.indices.index_manager.global_min_change_strategy import (
    UpdateIndexGlobalMinChange,
)
from anemone.indices.index_manager.node_exploration_manager import (
    NullNodeExplorationIndexManager,
)
from anemone.indices.node_indices.factory import create_exploration_index_data
from anemone.indices.node_indices.index_data import MinMaxPathValue
from anemone.node_evaluation.direct import NodeDirectEvaluator
from anemone.node_evaluation.direct.protocols import MasterStateValueEvaluator
from anemone.node_evaluation.full_refresh import iter_nodes
from anemone.node_evaluation.tree.single_agent.factory import NodeMaxEvaluationFactory
from anemone.node_evaluation.tree.single_agent.node_max_evaluation import (
    NodeMaxEvaluation,
)
from anemone.node_factory.algorithm_node_factory import AlgorithmNodeFactory
from anemone.node_factory.base import TreeNodeFactory
from anemone.node_selector.composed.composed_node_selector import ComposedNodeSelector
from anemone.node_selector.linoo import Linoo
from anemone.node_selector.opening_instructions import OpeningType
from anemone.node_selector.priority_check.no_priority_check import NoPriorityCheck
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.objectives import SingleAgentMaxObjective
from anemone.progress_monitor.progress_monitor import TreeBranchLimit
from anemone.rollouts import (
    RandomLegalPreferOpenableActionSelector,
    RandomOpenableActionSelector,
    RolloutOpeningExpansionExecutor,
)
from anemone.search_factory.search_factory import SearchFactory
from anemone.tree_exploration import TreeExploration
from anemone.tree_manager.opening_expansion_config import (
    OpeningExpansionKind,
    RolloutActionSelectorKind,
)
from anemone.tree_manager.opening_expansion_executor import (
    OnePlyOpeningExpansionExecutor,
)

from .contracts import (
    EvaluationIdentity,
    FrozenEvaluatorSource,
    PreparationLimits,
    PreparedEvaluatorContext,
    RuntimeInvalidated,
    codec_capabilities,
    require,
    validate_index_recipe,
)
from .native import (
    decode,
    derived,
    encode,
    fingerprint,
    reconstruct,
    validate_data_atoms,
    validate_index_payload,
)


def _owner_stamp(
    search: TreeExploration[Any], source: FrozenEvaluatorSource
) -> tuple[str, tuple[object, ...]]:
    """Validate actual native recipe and capture complete nonmutating preconditions."""
    require(
        type(search) is TreeExploration and not search._prepared_evaluator_invalidated,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=unidiomatic-typecheck,protected-access
        "unsupported or invalidated owner",
    )
    selector = search.node_selector
    require(
        type(selector) is ComposedNodeSelector  # pylint: disable=unidiomatic-typecheck
        and type(selector.base) is Linoo  # pylint: disable=unidiomatic-typecheck
        and type(selector.priority_check) is NoPriorityCheck,  # pylint: disable=unidiomatic-typecheck
        "unsupported selector recipe",
    )
    assert isinstance(selector, ComposedNodeSelector)
    base = cast("Linoo[Any]", selector.base)
    recipe = source.recipe
    validate_index_recipe(recipe)
    require(
        base.depth_selection_policy == recipe.depth_selection_policy,
        "selector recipe mismatch",
    )
    require(
        base.opening_instructor.opening_type == OpeningType.ALL_CHILDREN
        and base.opening_instructor.random_generator is base.random_generator,
        "opening/RNG recipe mismatch",
    )
    require(
        type(search.stopping_criterion) is TreeBranchLimit  # pylint: disable=unidiomatic-typecheck
        and search.stopping_criterion.tree_branch_limit == recipe.branch_limit,
        "stopping recipe mismatch",
    )
    manager = search.tree_manager
    factory = manager.tree_manager.node_factory
    require(
        type(factory) is AlgorithmNodeFactory,  # pylint: disable=unidiomatic-typecheck
        "unsupported node factory",
    )
    assert isinstance(factory, AlgorithmNodeFactory)
    require(
        type(factory.tree_node_factory) is TreeNodeFactory  # pylint: disable=unidiomatic-typecheck
        and factory.state_representation_factory is None
        and type(factory.node_tree_evaluation_factory) is NodeMaxEvaluationFactory,  # pylint: disable=unidiomatic-typecheck
        "unsupported factory/representation recipe",
    )
    # The callable slot accepts functions or methods. Inspect a bound method
    # only after the exact MethodType guard below.
    index_factory = cast("object", factory.exploration_index_data_create)
    require(
        type(index_factory) is MethodType  # pylint: disable=unidiomatic-typecheck
        and index_factory.__func__ is SearchFactory.node_index_create
        and type(index_factory.__self__) is SearchFactory  # pylint: disable=unidiomatic-typecheck
        and index_factory.__self__.index_computation is recipe.index_computation
        and index_factory.__self__.depth_index is False,
        "unsupported future-node index factory recipe",
    )
    evaluation_factory = cast(
        "NodeMaxEvaluationFactory[Any]", factory.node_tree_evaluation_factory
    )
    factory_policy = evaluation_factory.backup_policy
    require(
        evaluation_factory._evaluation_type is NodeMaxEvaluation  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
        and type(evaluation_factory.objective) is SingleAgentMaxObjective  # pylint: disable=unidiomatic-typecheck
        and type(factory_policy) is ExplicitMaxBackupPolicy  # pylint: disable=unidiomatic-typecheck
        and type(cast("object", factory_policy.aggregation_policy))  # pylint: disable=unidiomatic-typecheck
        is BestChildAggregationPolicy
        and type(factory_policy.proof_policy) is MaxProofPolicy,  # pylint: disable=unidiomatic-typecheck
        "unsupported future-node max/proof recipe",
    )
    require(
        type(manager.index_manager)  # pylint: disable=unidiomatic-typecheck
        is (
            NullNodeExplorationIndexManager
            if recipe.index_computation is None
            else UpdateIndexGlobalMinChange
        ),
        "unsupported index recipe",
    )
    require(
        manager.opening_expansion_config == recipe.opening_expansion,
        "opening expansion recipe mismatch",
    )
    executor = manager.opening_expansion_executor
    rollout_rng = rollout_random_generator(search)
    if recipe.opening_expansion.kind == OpeningExpansionKind.ONE_PLY:
        require(
            type(executor) is OnePlyOpeningExpansionExecutor  # pylint: disable=unidiomatic-typecheck
            and rollout_rng is None
            and recipe.rng_layout == "search_only",
            "unsupported one-ply/RNG layout",
        )
    else:
        require(
            type(executor) is RolloutOpeningExpansionExecutor,  # pylint: disable=unidiomatic-typecheck
            "unsupported rollout executor",
        )
        assert isinstance(executor, RolloutOpeningExpansionExecutor)
        allowed = {
            RolloutActionSelectorKind.RANDOM_OPENABLE: RandomOpenableActionSelector,
            RolloutActionSelectorKind.RANDOM_LEGAL_PREFER_OPENABLE: RandomLegalPreferOpenableActionSelector,
        }
        require(
            type(executor.rollout_action_selector)  # pylint: disable=unidiomatic-typecheck
            is allowed.get(recipe.opening_expansion.rollout.action_selector_kind),
            "unsupported rollout selector override",
        )
        require(
            executor.max_extra_steps == recipe.opening_expansion.rollout.max_extra_steps
            and executor.stop_on_existing_node
            == recipe.opening_expansion.rollout.stop_on_existing_node,
            "rollout executor recipe drift",
        )
        require(
            rollout_rng is not None
            and rollout_rng is not base.random_generator
            and recipe.rng_layout == "independent_rollout",
            "unsupported rollout RNG alias/ownership",
        )
    direct = manager.node_evaluator
    require(
        type(direct) is NodeDirectEvaluator  # pylint: disable=unidiomatic-typecheck
        and direct.current_evaluator_version == search.evaluator_version,
        "evaluator version coverage mismatch",
    )
    assert direct is not None
    require(
        getattr(direct.master_state_value_evaluator, "evaluation_identity", None)
        == source.identity.incumbent,
        "incumbent identity mismatch",
    )
    identities: list[object] = [
        id(search),
        id(search.tree),
        id(manager),
        id(manager.tree_manager),
        id(manager.index_manager),
        id(manager.value_propagator),
        id(factory),
        id(factory.tree_node_factory),
        id(evaluation_factory),
        id(factory.exploration_index_data_create),
        id(selector),
        id(base.random_generator),
        id(rollout_rng),
        id(executor),
        id(direct),
        id(direct.master_state_value_evaluator),
        id(search.latest_tree_expansions),
        id(search.latest_step_report),
    ]
    for node in iter_nodes(search.tree):
        require(
            type(node) is AlgorithmNode  # pylint: disable=unidiomatic-typecheck
            and node.state_representation is None
            and type(node.tree_evaluation) is NodeMaxEvaluation  # pylint: disable=unidiomatic-typecheck
            and (
                node.exploration_index_data is None
                if recipe.index_computation is None
                else type(node.exploration_index_data) is MinMaxPathValue  # pylint: disable=unidiomatic-typecheck
            ),
            "unsupported derived object type",
        )
        # The exact NodeMaxEvaluation check above has already rejected subclasses.
        evaluation = cast("NodeMaxEvaluation[Any]", node.tree_evaluation)
        policy = evaluation.backup_policy
        require(
            type(evaluation.required_objective) is SingleAgentMaxObjective  # pylint: disable=unidiomatic-typecheck
            and type(policy) is ExplicitMaxBackupPolicy  # pylint: disable=unidiomatic-typecheck
            and type(cast("object", policy.aggregation_policy))  # pylint: disable=unidiomatic-typecheck
            is BestChildAggregationPolicy
            and type(policy.proof_policy) is MaxProofPolicy,  # pylint: disable=unidiomatic-typecheck
            "unsupported max/proof recipe",
        )
        handle = node.state_handle
        require(
            type(handle) is CheckpointBackedStateHandle  # pylint: disable=unidiomatic-typecheck
            and type(handle.resolver.state_codec) is source._codec_type  # pyright: ignore[reportPrivateUsage]  # pylint: disable=unidiomatic-typecheck,protected-access
            and codec_capabilities(type(handle.resolver.state_codec))
            == source.codec_capability
            and getattr(handle.resolver.state_codec, "prepared_capabilities", None)
            == source.codec_capability,
            "owner codec/handle identity mismatch",
        )
        # require() is not a checker-recognized TypeGuard; admission already passed.
        handle = cast("CheckpointBackedStateHandle[Any]", handle)
        identities.append(
            (
                id(node),
                id(node.tree_node),
                id(handle),
                id(handle.resolver),
                id(evaluation),
                id(node.exploration_index_data),
            )
        )
    stamp = fingerprint(
        search.tree,
        base,
        search.latest_tree_expansions,
        search.evaluator_version,
        None if rollout_rng is None else rollout_rng.getstate(),
    )
    require(
        stamp == source.native_fingerprint,
        "stale head or incumbent/search/RNG baseline mismatch",
    )
    return stamp, tuple(identities)


@dataclass(frozen=True, slots=True)
class InstallReport:
    """Identity and exact installed coverage; no publication or acceptance receipt."""

    source: str
    candidate: EvaluationIdentity
    old_evaluator_version: int
    new_evaluator_version: int
    installed_nodes: int
    rng_objects_preserved: bool


def _assign_node(node: AlgorithmNode[Any], staged: AlgorithmNode[Any]) -> None:
    """Install only the staged evaluation and index, preserving all structure."""
    node.tree_evaluation = staged.tree_evaluation
    node.exploration_index_data = staged.exploration_index_data


class EvaluatorInstallPlan:
    """Single-use checked plan; caller holds the same exclusive quiescent ownership."""

    def __init__(
        self,
        search: TreeExploration[Any],
        context: PreparedEvaluatorContext,
        evaluator: MasterStateValueEvaluator,
        stamp: tuple[str, tuple[object, ...]],
        nodes: tuple[tuple[AlgorithmNode[Any], AlgorithmNode[Any]], ...],
        selector: ComposedNodeSelector[Any],
        limits: PreparationLimits,
    ) -> None:
        """Retain only staged bindings, bounded resources and a weak runtime owner."""
        self._owner = ref(search)
        self._context = context
        self._evaluator: MasterStateValueEvaluator | None = evaluator
        self._stamp = stamp
        self._nodes = nodes
        self._selector: ComposedNodeSelector[Any] | None = selector
        self._limits = limits
        self._used = False

    def dispose(self) -> None:
        """Release staged objects without touching the runtime or durable source."""
        self._used = True
        self._nodes = ()
        self._selector = None
        self._evaluator = None

    def install(self) -> InstallReport:
        """Recheck preconditions, then install once with no inference/backup/Growth."""
        require(not self._used, "installation plan already consumed")
        self._used = True
        try:
            self._limits.check()
            search = self._owner()
            require(search is not None, "installation owner no longer exists")
            assert search is not None
            context = self._context
            require(
                getattr(self._evaluator, "evaluation_identity", None)
                == context.evaluation_identity,
                "model identity changed after staging",
            )
            require(
                _owner_stamp(search, context.source) == self._stamp,
                "owner bindings changed after staging",
            )
            old_version = search.evaluator_version
            report = InstallReport(
                context.source.native_fingerprint,
                context.evaluation_identity,
                old_version,
                context.evaluator_version,
                len(self._nodes),
                True,
            )
            assert self._selector is not None and self._evaluator is not None
            self._limits.check()
        except BaseException:
            self.dispose()
            raise
        # From this point, any exception invalidates the private runtime. There
        # is intentionally no claim of general rollback or durable publication.
        search._prepared_evaluator_invalidated = True  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
        try:
            for node, staged in self._nodes:
                _assign_node(node, staged)
            search.node_selector = self._selector
            direct = search.tree_manager.node_evaluator
            assert direct is not None
            direct.master_state_value_evaluator = self._evaluator
            direct.current_evaluator_version = context.evaluator_version
            search.evaluator_version = context.evaluator_version
            search._prepared_evaluator_invalidated = False  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
        except BaseException as exc:
            self.dispose()
            message = "prepared installation failed; discard/reload runtime"
            raise RuntimeInvalidated(message) from exc
        self.dispose()
        return report


def stage_evaluator_context(
    search: TreeExploration[Any],
    *,
    current_source: FrozenEvaluatorSource,
    prepared: PreparedEvaluatorContext,
    evaluator: MasterStateValueEvaluator,
    evaluation_identity: EvaluationIdentity,
    limits: PreparationLimits,
) -> EvaluatorInstallPlan:
    """Allocate all candidate-derived objects and validate before owner mutation."""
    limits.check()
    require(
        len(current_source._checkpoint_bytes) <= limits.max_payload_bytes,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
        "source payload bound exceeded",
    )
    require(
        len(prepared._derived_bytes) <= limits.max_payload_bytes,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
        "derived payload bound exceeded",
    )
    require(prepared.source == current_source, "stale prepared source identity")
    require(
        prepared.evaluation_identity == evaluation_identity
        and getattr(evaluator, "evaluation_identity", None) == evaluation_identity,
        "model identity mismatch",
    )
    require(
        prepared.evaluator_version
        == decode(current_source._checkpoint_bytes).evaluator_version + 1,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
        "candidate evaluator version mismatch",
    )
    stamp = _owner_stamp(search, current_source)
    # Validate serialized context against a wholly independent arena. Restoration
    # uses native payload readers; candidate inference is never repeated here.
    arena = reconstruct(
        current_source._checkpoint_bytes,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
        current_source._codec_type,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
        current_source.recipe,
        limits,
        capability=current_source.codec_capability,
    )
    data = json.loads(prepared._derived_bytes)  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
    validate_data_atoms(data, limits)
    records = data["nodes"]
    owner_nodes = list(iter_nodes(search.tree))
    require(
        [r[0] for r in records] == [n.id for n in owner_nodes],
        "incomplete/incorrect ordered context coverage",
    )
    detached = {n.id: n for n in iter_nodes(arena.tree)}
    detached_state_ids = {id(n.state) for n in detached.values()}
    checked_resolvers: set[int] = set()
    for owner in owner_nodes:
        limits.check()
        handle = cast("CheckpointBackedStateHandle[Any]", owner.state_handle)
        if id(handle.resolver) in checked_resolvers:
            continue
        checked_resolvers.add(id(handle.resolver))
        require(
            not detached_state_ids.intersection(
                id(state)
                for state in handle.resolver._resolved_states.values()  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
            ),
            "decoded state aliases destination owner",
        )
    factory = NodeMaxEvaluationFactory[Any]()
    staged: list[tuple[AlgorithmNode[Any], AlgorithmNode[Any]]] = []
    for owner, (node_id, evaluation, index) in zip(owner_nodes, records, strict=True):
        limits.check()
        require(
            evaluation is not None,
            "missing derived coverage",
        )
        eval_payload = _node_evaluation_payload_from_mapping(evaluation)
        index_payload = (
            None if index is None else _exploration_index_payload_from_mapping(index)
        )
        validate_index_payload(index_payload, current_source.recipe)
        shell = AlgorithmNode(
            tree_node=owner.tree_node,
            tree_evaluation=factory.create(owner.tree_node),
            exploration_index_data=create_exploration_index_data(
                owner.tree_node, current_source.recipe.index_computation
            ),
            state_representation=None,
        )
        for target in (shell, detached[node_id]):
            _restore_evaluation(node=target, payload=eval_payload)
            _restore_exploration_index(node=target, payload=index_payload)
        staged.append((owner, shell))
    selector_payload = _selector_payload_from_mapping(data["selector"])
    objective = arena.tree.root_node.tree_evaluation.required_objective
    assert isinstance(objective, SingleAgentMaxObjective)
    require(
        arena.selector.restore_from_checkpoint_payload(
            tree=arena.tree,
            objective=objective,
            payload=selector_payload,
        ),
        "invalid prepared selector state",
    )
    require(
        encode(derived(arena.tree, arena.selector)) == prepared._derived_bytes,  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
        "derived payload failed exact native round trip",
    )
    # Built-in Linoo caches own ids, scalar signatures and counts, never node
    # references. Rebind its three native ownership handles to the staged owner.
    base = arena.selector
    owner_selector = cast("ComposedNodeSelector[Any]", search.node_selector)
    owner_base = cast("Linoo[Any]", owner_selector.base)
    base.random_generator = owner_base.random_generator
    base.opening_instructor = owner_base.opening_instructor
    base._cache_tree_identity = id(search.tree.root_node)  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
    root_staged = next(
        shell for owner, shell in staged if owner is search.tree.root_node
    )
    base._cache_objective_identity = id(root_staged.tree_evaluation.required_objective)  # pyright: ignore[reportPrivateUsage]  # pylint: disable=protected-access
    selector = ComposedNodeSelector(priority_check=NoPriorityCheck(), base=base)
    limits.check()
    require(
        _owner_stamp(search, current_source) == stamp, "owner changed during staging"
    )
    return EvaluatorInstallPlan(
        search, prepared, evaluator, stamp, tuple(staged), selector, limits
    )
