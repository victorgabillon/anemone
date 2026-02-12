"""Factory for constructing priority checks."""

from random import Random

from anemone.hooks.search_hooks import SearchHooks
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.node_selector.opening_instructions import OpeningInstructor
from anemone.node_selector.priority_check.args_union import PriorityCheckArgs
from anemone.node_selector.priority_check.no_priority_check import NoPriorityCheck
from anemone.node_selector.priority_check.priority_check import PriorityCheck
from anemone.node_selector.priority_check.registered_args import (
    RegisteredPriorityCheckArgs,
)


class UnknownPriorityCheckError(ValueError):
    """Raised when a priority-check type is not recognized."""

    def __init__(self, args: PriorityCheckArgs) -> None:
        """Initialize the error with a message indicating the unknown priority-check type."""
        super().__init__(
            f"priority check construction: can not find {args.type} {args} in file {__name__}"
        )


class MissingPriorityRegistryError(ValueError):
    """Raised when a registered priority check is requested without hooks."""

    def __init__(self) -> None:
        """Initialize the error for missing hook registry."""
        super().__init__(
            "PriorityRegistered requires hooks with a priority_check_registry."
        )


class UnknownRegisteredPriorityCheckError(KeyError):
    """Raised when a named registered priority check is absent."""

    def __init__(self, name: str) -> None:
        """Initialize the error for a missing registry entry."""
        super().__init__(f"Priority check '{name}' not found in registry.")


def create_priority_check(
    args: PriorityCheckArgs,
    random_generator: Random,
    hooks: SearchHooks | None,
    opening_instructor: OpeningInstructor,
) -> PriorityCheck:
    """Create a priority check from args."""
    match args.type:
        case NodeSelectorType.PRIORITY_NOOP:
            _ = random_generator
            _ = hooks
            _ = opening_instructor
            return NoPriorityCheck()
        case NodeSelectorType.PRIORITY_REGISTERED:
            assert isinstance(args, RegisteredPriorityCheckArgs)
            if hooks is None:
                raise MissingPriorityRegistryError
            factory = hooks.priority_check_registry.get(args.name)
            if factory is None:
                raise UnknownRegisteredPriorityCheckError(args.name)
            return factory(args.params, random_generator, hooks, opening_instructor)
        case _:
            raise UnknownPriorityCheckError(args)
