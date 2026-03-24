"""Provide the implementation of the SearchFactory abstract factory.

The SearchFactory class creates
dependent factories for selecting nodes to open and creating indices. These factories need to
operate on the same data, so they are created in a coherent way.

The SearchFactory class provides methods to create the node selector factory and
to create node indices for a given tree node.

Classes:
- SearchFactory: An abstract factory for creating dependent factories for selecting nodes to open, creating indices,
and exploring them coherently.

Protocols:
- SearchFactoryP: A protocol that defines the abstract methods for the SearchFactory class.

Functions:
- create_node_selector_factory: A function that creates the node selector factory.
- node_index_create: A function that creates node indices for a given tree node.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from random import Random
from typing import Any, Protocol

from valanga import State

from anemone import node_selector as node_selectors
from anemone import nodes
from anemone.dynamics import SearchDynamics
from anemone.hooks.search_hooks import SearchHooks
from anemone.indices import node_indices
from anemone.indices.node_indices.factory import (
    create_exploration_index_data,
)
from anemone.node_selector.composed.args import ComposedNodeSelectorArgs
from anemone.node_selector.factory import create_composed_node_selector
from anemone.node_selector.opening_instructions import (
    OpeningInstructor,
    OpeningType,
)
from anemone.node_selector.sequool.factory import (
    SequoolArgs,
)
from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode

NodeSelectorFactory = Callable[[], node_selectors.NodeSelector]


def _missing_selector_factory_dependency_error(
    dependency_name: str,
    *,
    details: str = "",
) -> ValueError:
    return ValueError(
        "SearchFactory.create_node_selector_factory() requires "
        f"`{dependency_name}`{details}."
    )


class SearchFactoryP(Protocol):
    """Create dependent factories for node selection and node-index creation.

    - the node selector
    - the index creator
    These collaborators need to operate on the same data, so they must be
    created coherently.
    """

    depth_index: bool

    def create_node_selector_factory(self) -> NodeSelectorFactory:
        """Create a NodeSelectorFactory object.

        Returns:
            NodeSelectorFactory: The created NodeSelectorFactory object.

        """
        ...

    def node_index_create[StateT: State](
        self,
        tree_node: nodes.TreeNode[AlgorithmNode[StateT], StateT],
    ) -> node_indices.NodeExplorationData[AlgorithmNode[StateT], StateT] | None:
        """Create a node index for the given tree node.

        Args:
            tree_node (nodes.TreeNode): The tree node for which to create the index.

        Returns:
            node_indices.NodeExplorationData | None: The created node index, or None if the index could not be created.

        """
        ...


def _base_selector_args(args: Any) -> Any:
    # args can be SequoolArgs / RecurZipfBaseArgs / UniformArgs / ComposedNodeSelectorArgs

    if isinstance(args, ComposedNodeSelectorArgs):
        return args.base
    return args


@dataclass
class SearchFactory:
    """Create dependent factories for node selection and node-index creation.

    - the node selector
    - the index creator
    These collaborators need to operate on the same data, so they must be
    created coherently.
    """

    node_selector_args: ComposedNodeSelectorArgs | None
    opening_type: OpeningType | None
    random_generator: Random | None
    dynamics: SearchDynamics[Any, Any]
    index_computation: node_indices.IndexComputationType | None
    hooks: SearchHooks | None = None
    depth_index: bool = False

    def __post_init__(self) -> None:
        """Initialize the object after it has been created.

        This method is automatically called after the object has been initialized.
        It sets the value of `depth_index` based on the type of `node_selector_args`.

        If `node_selector_args` is an instance of `SequoolArgs`, then `depth_index`
        is set to the value of `recursive_selection_on_all_nodes` attribute of `node_selector_args`.
        Otherwise, `depth_index` is set to False.
        """
        base_args = _base_selector_args(self.node_selector_args)
        if isinstance(base_args, SequoolArgs):
            self.depth_index = base_args.recursive_selection_on_all_nodes
        else:
            self.depth_index = False

    def create_node_selector_factory(self) -> NodeSelectorFactory:
        """Create the node selector factory.

        Returns:
            A callable object that creates the node selector.

        Raises:
            ValueError: If selector configuration is incomplete.

        """
        # creates the opening instructor

        if self.random_generator is None:
            raise _missing_selector_factory_dependency_error("random_generator")
        if self.node_selector_args is None:
            raise _missing_selector_factory_dependency_error("node_selector_args")
        if self.opening_type is None:
            raise _missing_selector_factory_dependency_error(
                "opening_type",
                details=" to build the OpeningInstructor",
            )

        opening_instructor = OpeningInstructor(
            opening_type=self.opening_type,
            random_generator=self.random_generator,
            dynamics=self.dynamics,
        )

        node_selector_create: NodeSelectorFactory = partial(
            create_composed_node_selector,
            args=self.node_selector_args,
            opening_instructor=opening_instructor,
            random_generator=self.random_generator,
            hooks=self.hooks,
        )
        return node_selector_create

    def node_index_create[StateT: State](
        self,
        tree_node: nodes.TreeNode[AlgorithmNode[StateT], StateT],
    ) -> node_indices.NodeExplorationData[AlgorithmNode[StateT], StateT] | None:
        """Create node indices for a given tree node.

        Args:
            tree_node: The tree node for which to create the node indices.

        Returns:
            An instance of the NodeExplorationData class if depth indexing is enabled, otherwise None.

        """
        exploration_index_data: (
            node_indices.NodeExplorationData[AlgorithmNode[StateT], StateT] | None
        ) = create_exploration_index_data(
            tree_node=tree_node,
            index_computation=self.index_computation,
            depth_index=self.depth_index,
        )

        return exploration_index_data
