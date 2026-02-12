"""Factory to build node selectors."""

from random import Random

from anemone.hooks.search_hooks import SearchHooks

from .all_node_selector_args import AllNodeSelectorArgs
from .composed.args import ComposedNodeSelectorArgs
from .composed.composed_node_selector import ComposedNodeSelector
from .node_selector import NodeSelector
from .node_selector_types import NodeSelectorType
from .opening_instructions import OpeningInstructor
from .priority_check.factory import create_priority_check
from .recurzipf.recur_zipf_base import RecurZipfBase, RecurZipfBaseArgs
from .sequool import SequoolArgs, create_sequool
from .uniform import Uniform


class UnknownNodeSelectorError(ValueError):
    """Raised when a node selector type is not recognized."""

    def __init__(self, args: AllNodeSelectorArgs) -> None:
        """Initialize the error with the unsupported selector arguments."""
        super().__init__(
            f"node selector construction: can not find {args.type}  {args} in file {__name__}"
        )


def create_composed_node_selector(
    args: ComposedNodeSelectorArgs,
    opening_instructor: OpeningInstructor,
    random_generator: Random,
    hooks: SearchHooks | None = None,
) -> ComposedNodeSelector:
    """Create a composed node selector from the given arguments."""
    priority_check = create_priority_check(
        args=args.priority,
        random_generator=random_generator,
        hooks=hooks,
    )
    base_selector = create(
        args=args.base,
        opening_instructor=opening_instructor,
        random_generator=random_generator,
        hooks=hooks,
    )
    return ComposedNodeSelector(
        priority_check=priority_check,
        base=base_selector,
    )


def create(
    args: AllNodeSelectorArgs,
    opening_instructor: OpeningInstructor,
    random_generator: Random,
    hooks: SearchHooks | None = None,
) -> NodeSelector:
    """Creation of a node selector."""
    node_branch_opening_selector: NodeSelector

    match args.type:
        case NodeSelectorType.UNIFORM:
            node_branch_opening_selector = Uniform(
                opening_instructor=opening_instructor
            )
        case NodeSelectorType.RECUR_ZIPF_BASE:
            assert isinstance(args, RecurZipfBaseArgs)
            node_branch_opening_selector = RecurZipfBase(
                args=args,
                random_generator=random_generator,
                opening_instructor=opening_instructor,
            )

        case NodeSelectorType.SEQUOOL:
            assert isinstance(args, SequoolArgs)
            node_branch_opening_selector = create_sequool(
                opening_instructor=opening_instructor,
                random_generator=random_generator,
                args=args,
            )

        case _:
            raise UnknownNodeSelectorError(args)

    return node_branch_opening_selector
