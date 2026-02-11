"""Union of all priority-check argument variants."""

from typing import TypeAlias, Union

from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs

PriorityCheckArgs: TypeAlias = Union[NoPriorityCheckArgs]
