"""Union of all priority-check argument variants."""

from anemone.node_selector.priority_check.noop_args import NoPriorityCheckArgs
from anemone.node_selector.priority_check.registered_args import (
    RegisteredPriorityCheckArgs,
)

PriorityCheckArgs = NoPriorityCheckArgs | RegisteredPriorityCheckArgs
