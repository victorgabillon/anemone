"""Error types and factories for the Linoo selector."""

from __future__ import annotations


class LinooSelectionError(ValueError):
    """Base error for invalid Linoo selection contexts."""


class LinooDirectValueUnavailableError(LinooSelectionError):
    """Raised when a frontier node does not expose a direct single-player value."""


class LinooIncompatibleObjectiveError(LinooSelectionError):
    """Raised when Linoo is used outside the single-player max setting."""


def no_frontier_nodes_error() -> LinooSelectionError:
    """Return the error for an exhausted Linoo frontier."""
    return LinooSelectionError(
        "Linoo could not find any unopened unresolved nodes in the current tree."
    )


def missing_root_objective_error() -> LinooIncompatibleObjectiveError:
    """Return the error for a missing root objective."""
    return LinooIncompatibleObjectiveError(
        "Linoo requires a configured SingleAgentMaxObjective on the tree root."
    )


def incompatible_objective_error(objective: object) -> LinooIncompatibleObjectiveError:
    """Return the error for a non-single-agent objective."""
    return LinooIncompatibleObjectiveError(
        f"Linoo requires a SingleAgentMaxObjective; got {type(objective).__name__}."
    )


def missing_direct_value_attribute_error(
    node_id: int,
) -> LinooDirectValueUnavailableError:
    """Return the error for an unavailable direct-value attribute."""
    return LinooDirectValueUnavailableError(
        "Linoo requires a direct single-player value on every frontier node; "
        f"node {node_id} does not expose tree_evaluation.direct_value."
    )


def missing_direct_value_error(node_id: int) -> LinooDirectValueUnavailableError:
    """Return the error for a missing direct value."""
    return LinooDirectValueUnavailableError(
        "Linoo requires a direct single-player value on every frontier node; "
        f"node {node_id} has no direct_value."
    )


def invalid_linoo_checkpoint_payload_error() -> ValueError:
    """Return the generic error for rejected optional selector state."""
    return ValueError("Invalid Linoo selector checkpoint payload.")


def invalid_linoo_depth_selection_policy_error(policy: object) -> ValueError:
    """Return the error for an unsupported Linoo depth-selection policy."""
    return ValueError(f"Invalid Linoo depth-selection policy: {policy!r}.")
