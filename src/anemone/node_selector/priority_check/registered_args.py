"""Arguments for selecting a registry-provided priority check."""

from dataclasses import dataclass, field
from typing import Any, Literal

from anemone.node_selector.node_selector_types import NodeSelectorType


def _registered_priority_check_args() -> dict[str, Any]:
    """Help to create a RegisteredPriorityCheckArgs dict."""
    return {}


@dataclass
class RegisteredPriorityCheckArgs:
    """Select a priority check by registry name from :class:`SearchHooks`."""

    type: Literal[NodeSelectorType.PRIORITY_REGISTERED]
    name: str
    params: dict[str, Any] = field(default_factory=_registered_priority_check_args)
