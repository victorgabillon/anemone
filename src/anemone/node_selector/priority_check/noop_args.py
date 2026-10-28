"""Arguments for the no-op priority check."""

from dataclasses import dataclass
from typing import Literal

from anemone.node_selector.node_selector_types import NodeSelectorType


@dataclass
class NoPriorityCheckArgs:
    """Args selecting the built-in no-op priority check."""

    type: Literal[NodeSelectorType.PRIORITY_NOOP]
