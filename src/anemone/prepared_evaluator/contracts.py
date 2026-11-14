"""Bounded identities and immutable artifacts for native prepared evaluation."""

# Codec admission deliberately rejects subclasses and custom metaclasses.
# Exact-type lint exceptions below preserve that safety boundary.

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal

from anemone.checkpoints._protocols import (
    IncrementalStateCheckpointCodec,  # noqa: TC001
)
from anemone.indices.node_indices.index_types import IndexComputationType
from anemone.tree_manager.opening_expansion_config import OpeningExpansionConfig


class PreparedEvaluatorRefusal(ValueError):  # noqa: N818
    """The source, configuration or complete installation proof is unsupported."""


class RuntimeInvalidated(RuntimeError):  # noqa: N818
    """Installation began and failed; discard/reload the private runtime."""


def require(condition: bool, reason: str) -> None:
    """Refuse without substituting defaults or weakening comparison."""
    if not condition:
        raise PreparedEvaluatorRefusal(reason)


@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    """Complete opaque prediction identity; authentication remains with the caller."""

    bundle: str
    weights: str
    architecture: str
    encoder: str
    state_input: str
    objective: str
    numerical_profile: Literal["cpu-fp32-strict-scalar-v1"] = (
        "cpu-fp32-strict-scalar-v1"
    )


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Caller-authenticated publication, incumbent and codec/build identity."""

    publication: str
    incumbent: str | EvaluationIdentity
    codec: str
    anemone_build: str
    domain: str


@dataclass(frozen=True, slots=True)
class OwnedInlineCodecCapabilities:
    """Trusted codec promise: fresh owned states from inert inline payloads only.

    Schema identity is stable across instances and changes with decoding semantics.
    No external stores, leases, live objects or shared mutable state are permitted.
    This declaration is not a sandbox for arbitrary Python codec implementations.
    """

    schema_identity: str
    reconstruction: Literal["owned-inline-v1"] = "owned-inline-v1"


def codec_capabilities(
    codec_type: type[IncrementalStateCheckpointCodec[Any]],
) -> OwnedInlineCodecCapabilities:
    """Admit a concrete fresh-constructible class, never a supplied instance/factory."""
    require(
        type(codec_type) is type and codec_type.__new__ is object.__new__,  # pylint: disable=unidiomatic-typecheck
        "codec must be a fresh zero-argument class, not a factory or singleton",
    )
    capability = getattr(codec_type, "prepared_capabilities", None)
    require(
        type(capability) is OwnedInlineCodecCapabilities,  # pylint: disable=unidiomatic-typecheck
        "missing owned-inline codec capability",
    )
    assert isinstance(capability, OwnedInlineCodecCapabilities)
    require(
        type(capability.schema_identity) is str  # pylint: disable=unidiomatic-typecheck
        and bool(capability.schema_identity.strip())
        and capability.reconstruction == "owned-inline-v1",
        "unsupported codec schema/capability; external resources unsupported",
    )
    return capability


@dataclass(frozen=True, slots=True)
class SupportedSearchRecipe:
    """Finite v1 recipe: max/proof, null or global-min-change, noop/Linoo."""

    depth_selection_policy: str = "alternating_by_step"
    branch_limit: int = 4096
    opening_expansion: OpeningExpansionConfig = field(
        default_factory=OpeningExpansionConfig
    )
    rng_layout: Literal["search_only", "independent_rollout"] = "search_only"
    index_computation: Literal[IndexComputationType.MIN_GLOBAL_CHANGE] | None = (
        IndexComputationType.MIN_GLOBAL_CHANGE
    )


def validate_index_recipe(recipe: SupportedSearchRecipe) -> None:
    """Admit only the two canonical recipes, never string aliases or defaults."""
    require(
        recipe.index_computation is None
        or recipe.index_computation is IndexComputationType.MIN_GLOBAL_CHANGE,
        "unsupported index recipe",
    )


@dataclass(frozen=True, slots=True)
class PreparationLimits:
    """Cooperative allocation/work bounds; caller retains hard process containment."""

    deadline_monotonic: float
    max_nodes: int = 512
    max_branches: int = 4096
    max_payload_bytes: int = 16 * 1024 * 1024
    max_state_decodes: int = 1024
    cancelled: Callable[[], bool] | None = None

    def check(self) -> None:
        """Check the original deadline; no phase renews it."""
        require(self.cancelled is None or not self.cancelled(), "preparation cancelled")
        require(monotonic() < self.deadline_monotonic, "preparation deadline exhausted")
        require(
            min(
                self.max_nodes,
                self.max_branches,
                self.max_payload_bytes,
                self.max_state_decodes,
            )
            > 0,
            "preparation bounds must be positive",
        )


@dataclass(frozen=True, slots=True)
class FrozenEvaluatorSource:
    """Owned immutable native records, with no live node/resolver/selector/RNG."""

    identity: SourceIdentity
    recipe: SupportedSearchRecipe
    checkpoint_sha256: str
    native_fingerprint: str
    _checkpoint_bytes: bytes
    _codec_type: type[IncrementalStateCheckpointCodec[Any]]
    codec_capability: OwnedInlineCodecCapabilities


@dataclass(frozen=True, slots=True)
class PreparedEvaluatorContext:
    """Immutable derived state only; this is not a checkpoint publication API."""

    source: FrozenEvaluatorSource
    evaluation_identity: EvaluationIdentity
    evaluator_version: int
    requested_count: int
    reevaluated_count: int
    changed_count: int
    skipped_terminal_count: int
    skipped_certified_count: int
    _derived_bytes: bytes
