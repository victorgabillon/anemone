"""Arguments for composed node selector."""

from dataclasses import dataclass
from typing import Literal

from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.priority_check.args_union import PriorityCheckArgs

from anemone.node_selector.all_node_selector_args import AllNodeSelectorArgs


@dataclass
class ComposedNodeSelectorArgs:
    """Compose a priority check with a base node selector."""

    type: Literal[NodeSelectorType.COMPOSED]
    priority: PriorityCheckArgs
    base: AllNodeSelectorArgs
