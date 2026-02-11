"""Arguments for composed node selector."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.priority_check.args_union import PriorityCheckArgs

if TYPE_CHECKING:
    from anemone.node_selector.factory import AllNodeSelectorArgs


@dataclass
class ComposedNodeSelectorArgs:
    """Compose a priority check with a base node selector."""

    type: Literal[NodeSelectorType.COMPOSED]
    priority: PriorityCheckArgs
    base: "AllNodeSelectorArgs"
