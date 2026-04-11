"""Builders for training-oriented search-tree export."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from .model import (
    TRAINING_TREE_SNAPSHOT_FORMAT_KIND,
    TRAINING_TREE_SNAPSHOT_FORMAT_VERSION,
    TrainingNodeSnapshot,
    TrainingTreeSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class StateRefDumper(Protocol):
    """Protocol for caller-supplied reversible state-reference dumping."""

    def __call__(self, state: object) -> Any:
        """Return an opaque dumped payload for one state."""
        ...


class ValueScalarExtractor(Protocol):
    """Protocol for caller-supplied raw-scalar extraction from values."""

    def __call__(self, value: object | None) -> float | None:
        """Return a numeric scalar for one value-like object."""
        ...


def build_training_node_snapshot(
    node: object,
    *,
    state_ref_dumper: StateRefDumper | None = None,
    direct_value_extractor: ValueScalarExtractor | None = None,
    backed_up_value_extractor: ValueScalarExtractor | None = None,
) -> TrainingNodeSnapshot:
    """Build one training-grade node snapshot from a live or dummy node."""
    direct_value = _get_direct_value(node)
    backed_up_value = _get_backed_up_value(node)
    return TrainingNodeSnapshot(
        node_id=_get_node_id(node),
        parent_ids=_get_parent_ids(node),
        child_ids=_get_child_ids(node),
        depth=_get_depth(node),
        state_ref_payload=_dump_state_ref(node, state_ref_dumper=state_ref_dumper),
        direct_value_scalar=_extract_value_scalar(
            direct_value,
            extractor=direct_value_extractor,
        ),
        backed_up_value_scalar=_extract_value_scalar(
            backed_up_value,
            extractor=backed_up_value_extractor,
        ),
        is_terminal=_get_is_terminal(node),
        is_exact=_get_is_exact(node),
        over_event_label=_get_over_event_label(node),
        visit_count=_get_visit_count(node),
        metadata=_get_metadata(node),
    )


def build_training_tree_snapshot(
    nodes: Sequence[object],
    *,
    root_node_id: str | None = None,
    state_ref_dumper: StateRefDumper | None = None,
    direct_value_extractor: ValueScalarExtractor | None = None,
    backed_up_value_extractor: ValueScalarExtractor | None = None,
    created_at_unix_s: float | None = None,
    metadata: dict[str, object] | None = None,
) -> TrainingTreeSnapshot:
    """Build one training-grade tree snapshot in the caller-provided node order."""
    node_snapshots = tuple(
        build_training_node_snapshot(
            node,
            state_ref_dumper=state_ref_dumper,
            direct_value_extractor=direct_value_extractor,
            backed_up_value_extractor=backed_up_value_extractor,
        )
        for node in nodes
    )
    return TrainingTreeSnapshot(
        root_node_id=_resolve_root_node_id(root_node_id, node_snapshots),
        nodes=node_snapshots,
        created_at_unix_s=created_at_unix_s,
        metadata=_build_tree_metadata(metadata),
    )


def _safe_getattr(obj: object, attribute_name: str) -> object | None:
    """Return ``obj.attribute_name`` when available."""
    try:
        return cast("object", getattr(obj, attribute_name))
    except (AttributeError, TypeError):
        return None


def _call_optional_no_arg(callable_candidate: object | None) -> object | None:
    """Call a no-arg callable candidate when possible."""
    if callable_candidate is None:
        return None
    if not callable(callable_candidate):
        return callable_candidate
    try:
        return cast("Callable[[], object]", callable_candidate)()
    except (AttributeError, TypeError):
        return None


def _normalize_node_id(value: object) -> str:
    """Return a stable string node id from one node-like object or raw id."""
    node_id = _safe_getattr(value, "node_id")
    if node_id is not None:
        return str(node_id)

    generic_id = _safe_getattr(value, "id")
    if generic_id is not None:
        return str(generic_id)

    return str(value)


def _get_node_id(node: object) -> str:
    """Return the node id as a stable string."""
    return _normalize_node_id(node)


def _get_parent_ids(node: object) -> tuple[str, ...]:
    """Return stable parent ids from direct ids or parent-node mappings."""
    parent_ids = _safe_getattr(node, "parent_ids")
    if isinstance(parent_ids, tuple | list):
        return tuple(_normalize_node_id(parent_id) for parent_id in parent_ids)
    if isinstance(parent_ids, set | frozenset):
        return tuple(sorted(_normalize_node_id(parent_id) for parent_id in parent_ids))

    parent_nodes = _safe_getattr(node, "parent_nodes")
    if isinstance(parent_nodes, Mapping):
        return tuple(sorted(_normalize_node_id(parent) for parent in parent_nodes))

    return ()


def _get_child_ids(node: object) -> tuple[str, ...]:
    """Return stable child ids from direct ids or branch-to-child mappings."""
    child_ids = _safe_getattr(node, "child_ids")
    if isinstance(child_ids, tuple | list):
        return tuple(_normalize_node_id(child_id) for child_id in child_ids)
    if isinstance(child_ids, set | frozenset):
        return tuple(sorted(_normalize_node_id(child_id) for child_id in child_ids))

    branches_children = _safe_getattr(node, "branches_children")
    if not isinstance(branches_children, Mapping):
        return ()

    unique_child_ids = {
        _normalize_node_id(child)
        for child in branches_children.values()
        if child is not None
    }
    return tuple(sorted(unique_child_ids))


def _get_depth(node: object) -> int:
    """Return the node depth or ``0`` when unavailable."""
    depth = _safe_getattr(node, "depth")
    if depth is None:
        depth = _safe_getattr(node, "tree_depth")
    if depth is None:
        return 0
    return _coerce_int(depth, default=0)


def _get_state(node: object) -> object | None:
    """Return the node state when available."""
    return _safe_getattr(node, "state")


def _dump_state_ref(
    node: object,
    *,
    state_ref_dumper: StateRefDumper | None,
) -> Any | None:
    """Return a caller-dumped reversible state payload when possible."""
    if state_ref_dumper is None:
        return None

    state = _get_state(node)
    if state is None:
        return None
    return state_ref_dumper(state)


def _get_tree_evaluation(node: object) -> object | None:
    """Return the tree-evaluation-like object when present."""
    evaluation = _safe_getattr(node, "tree_evaluation")
    if evaluation is not None:
        return evaluation
    return _safe_getattr(node, "evaluation")


def _get_direct_value(node: object) -> object | None:
    """Return the direct value-like object when present."""
    evaluation = _get_tree_evaluation(node)
    if evaluation is not None:
        direct_value = _safe_getattr(evaluation, "direct_value")
        if direct_value is not None:
            return direct_value
    return _safe_getattr(node, "direct_value")


def _get_backed_up_value(node: object) -> object | None:
    """Return the backed-up value-like object when present."""
    evaluation = _get_tree_evaluation(node)
    if evaluation is not None:
        backed_up_value = _safe_getattr(evaluation, "backed_up_value")
        if backed_up_value is not None:
            return backed_up_value

        minmax_value = _safe_getattr(evaluation, "minmax_value")
        if minmax_value is not None:
            return minmax_value

    backed_up_value = _safe_getattr(node, "backed_up_value")
    if backed_up_value is not None:
        return backed_up_value
    return _safe_getattr(node, "minmax_value")


def _extract_value_scalar(
    value: object | None,
    *,
    extractor: ValueScalarExtractor | None,
) -> float | None:
    """Return a caller-provided raw scalar for one value-like object."""
    if extractor is None:
        return None
    return extractor(value)


def _get_is_terminal(node: object) -> bool:
    """Return whether the node exposes terminal semantics."""
    direct_flag = _coerce_optional_bool(_safe_getattr(node, "is_terminal"))
    if direct_flag is not None:
        return direct_flag

    over_flag = _coerce_optional_bool(_safe_getattr(node, "is_over"))
    if over_flag is not None:
        return over_flag

    evaluation = _get_tree_evaluation(node)
    if evaluation is not None:
        evaluation_flag = _coerce_optional_bool(_safe_getattr(evaluation, "is_terminal"))
        if evaluation_flag is not None:
            return evaluation_flag

    return _value_is_terminal(_get_value_candidate(node))


def _get_is_exact(node: object) -> bool:
    """Return whether the node exposes exact-value semantics."""
    direct_flag = _coerce_optional_bool(_safe_getattr(node, "is_exact"))
    if direct_flag is not None:
        return direct_flag

    evaluation = _get_tree_evaluation(node)
    if evaluation is not None:
        evaluation_flag = _coerce_optional_bool(
            _safe_getattr(evaluation, "has_exact_value")
        )
        if evaluation_flag is not None:
            return evaluation_flag

        evaluation_flag = _coerce_optional_bool(_safe_getattr(evaluation, "is_exact"))
        if evaluation_flag is not None:
            return evaluation_flag

    return _value_is_exact(_get_value_candidate(node))


def _get_over_event_label(node: object) -> str | None:
    """Return a string over-event marker when present."""
    direct_over_event = _safe_getattr(node, "over_event")
    if direct_over_event is not None:
        return _format_over_event_label(direct_over_event)

    evaluation = _get_tree_evaluation(node)
    if evaluation is not None:
        evaluation_over_event = _safe_getattr(evaluation, "over_event")
        if evaluation_over_event is not None:
            return _format_over_event_label(evaluation_over_event)

        over_event_candidate = _call_optional_no_arg(
            _safe_getattr(evaluation, "get_over_event_candidate")
        )
        if over_event_candidate is not None:
            return _format_over_event_label(over_event_candidate)

    value = _get_value_candidate(node)
    if value is None:
        return None
    return _format_over_event_label(_safe_getattr(value, "over_event"))


def _get_visit_count(node: object) -> int | None:
    """Return a visit count when the node exposes one."""
    for attribute_name in ("visit_count", "visits", "n_visits", "visit_counter"):
        value = _safe_getattr(node, attribute_name)
        if value is None or isinstance(value, bool):
            continue
        return _coerce_int(value)
    return None


def _get_metadata(node: object) -> dict[str, Any]:
    """Return a shallow-copied metadata mapping when available."""
    metadata = _safe_getattr(node, "metadata")
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def _coerce_optional_bool(value: object | None) -> bool | None:
    """Return a bool from one optional attribute or no-arg callable."""
    resolved = _call_optional_no_arg(value)
    if resolved is None:
        return None
    return bool(resolved)


def _get_value_candidate(node: object) -> object | None:
    """Return the best available value candidate for exactness fallbacks."""
    backed_up_value = _get_backed_up_value(node)
    if backed_up_value is not None:
        return backed_up_value
    return _get_direct_value(node)


def _value_is_exact(value: object | None) -> bool:
    """Return whether a value-like object is exact from certainty metadata."""
    certainty_name = _certainty_name(value)
    return certainty_name in {"FORCED", "TERMINAL"}


def _value_is_terminal(value: object | None) -> bool:
    """Return whether a value-like object is terminal from certainty metadata."""
    return _certainty_name(value) == "TERMINAL"


def _certainty_name(value: object | None) -> str | None:
    """Return a normalized certainty name from one value-like object."""
    if value is None:
        return None

    certainty = _safe_getattr(value, "certainty")
    if certainty is None:
        return None

    certainty_name = _safe_getattr(certainty, "name")
    normalized = certainty_name if certainty_name is not None else certainty
    return str(normalized).upper()


def _format_over_event_label(over_event: object | None) -> str | None:
    """Return a compact string label for an over-event-like object."""
    if over_event is None:
        return None

    get_over_tag = _safe_getattr(over_event, "get_over_tag")
    if callable(get_over_tag):
        try:
            return str(get_over_tag())
        except (AttributeError, TypeError):
            return str(over_event)

    return str(over_event)


def _resolve_root_node_id(
    root_node_id: str | None,
    node_snapshots: tuple[TrainingNodeSnapshot, ...],
) -> str | None:
    """Return the explicit root id or derive one from root-like nodes."""
    if root_node_id is not None:
        return str(root_node_id)
    for node_snapshot in node_snapshots:
        if not node_snapshot.parent_ids:
            return node_snapshot.node_id
    if not node_snapshots:
        return None
    return node_snapshots[0].node_id


def _build_tree_metadata(
    metadata: dict[str, object] | None,
) -> dict[str, Any]:
    """Return snapshot metadata enriched with format identity."""
    built_metadata: dict[str, Any] = {
        "format_kind": TRAINING_TREE_SNAPSHOT_FORMAT_KIND,
        "format_version": TRAINING_TREE_SNAPSHOT_FORMAT_VERSION,
    }
    if metadata is not None:
        built_metadata.update(metadata)
    return built_metadata


def _coerce_int(value: object, *, default: int | None = None) -> int:
    """Return ``value`` as an ``int`` for supported scalar payloads."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | str):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if default is not None:
        return default
    raise TypeError


__all__ = [
    "StateRefDumper",
    "ValueScalarExtractor",
    "build_training_node_snapshot",
    "build_training_tree_snapshot",
]
