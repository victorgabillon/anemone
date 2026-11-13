"""Freeze and prepare immutable evaluation artifacts, never live Growth objects."""

# ruff: noqa: TC001
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from hashlib import sha256
from typing import Any

from anemone.checkpoints._protocols import IncrementalStateCheckpointCodec
from anemone.checkpoints.io import _encode_checkpoint_payload
from anemone.checkpoints.payloads import SearchRuntimeCheckpointPayload
from anemone.indices.index_manager.factory import create_exploration_index_manager
from anemone.indices.node_indices.index_types import IndexComputationType
from anemone.node_evaluation.direct import EvaluationQueries, create_node_evaluator
from anemone.node_evaluation.direct.protocols import MasterStateValueEvaluator
from anemone.node_evaluation.full_refresh import refresh_bound_context

from .contracts import (
    EvaluationIdentity,
    FrozenEvaluatorSource,
    PreparationLimits,
    PreparedEvaluatorContext,
    SourceIdentity,
    SupportedSearchRecipe,
    require,
)
from .native import (
    derived,
    encode,
    fingerprint,
    preflight,
    reconstruct,
    validate_source,
)


def _ambient_rng() -> object:
    """Observe already-loaded framework RNGs without importing optional frameworks."""
    values: list[object] = [random.getstate()]
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        state = numpy.random.get_state()
        values.append((state[0], state[1].tobytes(), *state[2:]))
    torch = sys.modules.get("torch")
    if torch is not None:
        require(not torch.cuda.is_initialized(), "prepared v1 refuses initialized CUDA")
        values.append(bytes(torch.random.get_rng_state().tolist()))
    return values


def freeze_evaluator_source(
    checkpoint: SearchRuntimeCheckpointPayload,
    *,
    codec_type: type[IncrementalStateCheckpointCodec[Any]],
    identity: SourceIdentity,
    recipe: SupportedSearchRecipe,
    limits: PreparationLimits,
) -> FrozenEvaluatorSource:
    """Pin a closed native payload by value and prove its cold-normalized identity.

    V1 accepts explicitly admitted zero-argument inline Morpion codecs. Their
    ``prepared_domain`` class declaration is ``morpion-inline-v1`` and promises
    fresh independently owned decoded state with no external stores/resources.
    External store/lease factories and uninitialized Linoo histories refuse.
    The caller authenticates publication/build/codec identities before this call.
    """
    require(
        all(asdict(identity).values()) and identity.domain == "morpion",
        "incomplete source identity",
    )
    validate_source(checkpoint, recipe, limits)
    raw = _encode_checkpoint_payload(checkpoint).json_bytes
    require(len(raw) <= limits.max_payload_bytes, "source payload bound exceeded")
    before = _ambient_rng()
    arena = reconstruct(raw, codec_type, recipe, limits)
    stamp = fingerprint(
        arena.tree,
        arena.selector,
        arena.expansions,
        checkpoint.evaluator_version,
        arena.payload.rollout_rng_state,
    )
    require(_ambient_rng() == before, "codec reconstruction consumed ambient RNG")
    limits.check()
    return FrozenEvaluatorSource(
        identity, recipe, sha256(raw).hexdigest(), stamp, raw, codec_type
    )


class _BoundedEvaluator:
    """Preserve scalar inference order while checking cooperative batch deadlines."""

    def __init__(
        self, evaluator: MasterStateValueEvaluator, limits: PreparationLimits
    ) -> None:
        self.evaluator = evaluator
        self.limits = limits
        self.over = evaluator.over

    def evaluate(self, state: Any) -> Any:
        """Delegate scalar inference with the same deadline checks."""
        self.limits.check()
        result = self.evaluator.evaluate(state)
        self.limits.check()
        return result

    def evaluate_batch_items(self, items: Sequence[Any]) -> Any:
        self.limits.check()
        result = self.evaluator.evaluate_batch_items(items)
        self.limits.check()
        return result


def prepare_evaluator_context(
    source: FrozenEvaluatorSource,
    *,
    evaluator: MasterStateValueEvaluator,
    evaluation_identity: EvaluationIdentity,
    limits: PreparationLimits,
    validate_direct_certificate: Callable[[Any], bool] | None = None,
) -> PreparedEvaluatorContext:
    """Run the single canonical full-refresh kernel in fresh detached ownership.

    No old-head backups/indexes/selector caches are reusable for catch-up. Freeze
    the complete new incumbent head and prepare again. An external evaluator may
    reuse raw predictions only under its full identity and exact input contract.
    """
    limits.check()
    require(
        all(asdict(evaluation_identity).values())
        and evaluation_identity.numerical_profile == "cpu-fp32-strict-scalar-v1",
        "unsupported/incomplete evaluation identity",
    )
    require(
        getattr(evaluator, "evaluation_identity", None) == evaluation_identity,
        "model identity mismatch",
    )
    require(
        sha256(source._checkpoint_bytes).hexdigest() == source.checkpoint_sha256,
        "source bytes changed",
    )
    before = _ambient_rng()
    arena = reconstruct(
        source._checkpoint_bytes, source._codec_type, source.recipe, limits
    )
    require(
        fingerprint(
            arena.tree,
            arena.selector,
            arena.expansions,
            arena.payload.evaluator_version,
            arena.payload.rollout_rng_state,
        )
        == source.native_fingerprint,
        "source native identity changed",
    )
    preflight(arena, validate_direct_certificate, limits)
    rng = arena.random.getstate()
    version = arena.payload.evaluator_version + 1
    direct = create_node_evaluator(_BoundedEvaluator(evaluator, limits))
    direct.current_evaluator_version = version
    report = refresh_bound_context(
        tree=arena.tree,
        evaluator=direct,
        evaluation_queries=EvaluationQueries(),
        evaluator_version=version,
        index_manager=create_exploration_index_manager(
            IndexComputationType.MIN_GLOBAL_CHANGE
        ),
        node_selector=arena.selector,
        latest_tree_expansions=arena.expansions,
        batch_size=64,
        validate_direct_certificate=validate_direct_certificate,
    )
    require(
        arena.random.getstate() == rng and _ambient_rng() == before,
        "preparation consumed RNG",
    )
    raw = encode(derived(arena.tree, arena.selector))
    require(len(raw) <= limits.max_payload_bytes, "derived payload bound exceeded")
    limits.check()
    return PreparedEvaluatorContext(
        source,
        evaluation_identity,
        version,
        report.requested_count,
        report.reevaluated_count,
        report.changed_count,
        report.skipped_terminal_count,
        report.skipped_certified_count,
        raw,
    )
