"""Factory for constructing priority checks."""

from random import Random

from anemone.hooks.search_hooks import SearchHooks
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.priority_check.args_union import PriorityCheckArgs
from anemone.node_selector.priority_check.no_priority_check import NoPriorityCheck
from anemone.node_selector.priority_check.priority_check import PriorityCheck


class UnknownPriorityCheckError(ValueError):
    """Raised when a priority-check type is not recognized."""

    def __init__(self, args: PriorityCheckArgs) -> None:
        """Initialize the error with a message indicating the unknown priority-check type."""
        super().__init__(
            f"priority check construction: can not find {args.type} {args} in file {__name__}"
        )


def create_priority_check(
    args: PriorityCheckArgs,
    random_generator: Random,
    hooks: SearchHooks | None,
) -> PriorityCheck:
    """Create a priority check from args."""
    _ = random_generator
    _ = hooks

    match args.type:
        case NodeSelectorType.PRIORITY_NOOP:
            return NoPriorityCheck()
        case _:
            raise UnknownPriorityCheckError(args)
