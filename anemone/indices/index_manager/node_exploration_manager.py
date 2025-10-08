"""
Module that contains the logic to compute the exploration index of a node in a tree.
"""

import math
from typing import TYPE_CHECKING, Protocol, Any


from anemone.trees.descendants import RangedDescendants
from anemone.trees import ValueTree
from anemone.basics import BranchKey
from valanga import Colors, HasTurn
from anemone.indices.node_indices.index_data import (
    IntervalExplo,
    MinMaxPathValue, 
    RecurZipfQuoolExplorationData,
    NodeExplorationData,
)
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from anemone.utils.small_tools import (
    Interval,
    distance_number_to_interval,
    intersect_intervals,
)

if TYPE_CHECKING:
    from anemone.nodes.itree_node import ITreeNode


class NodeExplorationIndexManager[T_NodeExplorationData: NodeExplorationData= Any, Content=Any](Protocol):
    """
    A protocol for managing the exploration indices of nodes in a tree.

    This protocol defines methods for updating the exploration indices of nodes in a tree.

    Args:
        Protocol (type): The base protocol type.
    """

    def update_root_node_index(
        self,
        root_node: AlgorithmNode,
        root_node_exploration_index_data: T_NodeExplorationData| None,
    ) -> None:
        """
        Updates the exploration index of the root node in the tree.

        Args:
            root_node (AlgorithmNode): The root node of the tree.
        """
        ...

    def update_node_indices(
        self,
        child_node: AlgorithmNode,
        parent_node: AlgorithmNode,
        parent_node_exploration_index_data: T_NodeExplorationData | None,
        child_node_exploration_index_data: T_NodeExplorationData | None,
        parent_node_content: Content,
        tree: ValueTree,
        child_rank: int,
    ) -> None:
        """
        Updates the exploration index of a child node in the tree.

        Args:
            child_node (AlgorithmNode): The child node to update.
            parent_node (AlgorithmNode): The parent node of the child node.
            tree (trees.MoveAndValueTree): The tree containing the nodes.
            child_rank (int): The rank of the child node among its siblings.
        """
        ...


class NullNodeExplorationIndexManager(NodeExplorationIndexManager):
    """
    A class representing a null node exploration index manager.

    This class is used when there is no need to update the exploration index of nodes in a tree.
    It inherits from the NodeExplorationIndexManager class.
    """

    def update_root_node_index(
        self,
        root_node: AlgorithmNode,
        root_node_exploration_index_data: NodeExplorationData | None ,
    ) -> None:
        """
        Updates the exploration index of the root node in the tree.

        Args:
            root_node (AlgorithmNode): The root node of the tree.
        """
        ...

    def update_node_indices(
        self,
        child_node: AlgorithmNode,
        parent_node: AlgorithmNode,
        parent_node_exploration_index_data: NodeExplorationData | None,
        child_node_exploration_index_data: NodeExplorationData | None,
        parent_node_content: None,
        tree: ValueTree,
        child_rank: int,
    ) -> None:
        """
        Updates the exploration index of a child node in the tree.

        Args:
            child_node (AlgorithmNode): The child node to update.
            parent_node (AlgorithmNode): The parent node of the child node.
            tree (trees.MoveAndValueTree): The tree containing the nodes.
            child_rank (int): The rank of the child node among its siblings.
        """
        raise Exception("should not be raised")


class UpdateIndexGlobalMinChange:
    """
    A class that updates the exploration index of nodes in a tree using the global minimum change strategy.
    """

    def update_root_node_index(
        self,
        root_node: AlgorithmNode,
        root_node_exploration_index_data: MinMaxPathValue |None,
    ) -> None:
        """
        Updates the exploration index of the root node in the tree using the global minimum change strategy.

        Args:
            root_node (AlgorithmNode): The root node of the tree.
        """
        root_value: float = root_node.minmax_evaluation.get_value_white()

        assert root_node_exploration_index_data is not None

        root_node_exploration_index_data.min_path_value = root_value
        root_node_exploration_index_data.max_path_value = root_value
        root_node_exploration_index_data.index = 0

    def update_node_indices(
        self,
        child_node: AlgorithmNode,
        parent_node: AlgorithmNode,
        parent_node_exploration_index_data: MinMaxPathValue |None,
        child_node_exploration_index_data: MinMaxPathValue | None,
        parent_node_content:None,
        tree: ValueTree,
        child_rank: int,
    ) -> None:
        """
        Updates the exploration index of a child node in the tree using the global minimum change strategy.

        Args:
            child_node (AlgorithmNode): The child node to update.
            parent_node (AlgorithmNode): The parent node of the child node.
            tree (trees.MoveAndValueTree): The tree containing the nodes.
            child_rank (int): The rank of the child node among its siblings.
        """

        assert parent_node_exploration_index_data is not None
        assert parent_node_exploration_index_data.min_path_value is not None
        assert parent_node_exploration_index_data.max_path_value is not None
        assert child_node_exploration_index_data is not None

        child_value: float = child_node.minmax_evaluation.get_value_white()

        child_node_min_path_value = min(
            child_value, parent_node_exploration_index_data.min_path_value
        )

        child_node_max_path_value = max(
            child_value, parent_node_exploration_index_data.max_path_value
        )

        # computes local_child_index the amount of change for the child node to become better than its parent
        child_index: float = (
            abs(child_node_max_path_value - child_node_min_path_value) / 2
        )

        # the amount of change for the child to become better than any of its ancestor
        # and become the overall best bode, the max is computed with the parent index
        # child_index: float = max(local_child_index, parent_index)

        # the index of the child node is updated now
        # as a child node can have multiple parents we take the min if an index was previously computed
        if child_node_exploration_index_data.index is None:
            child_node_exploration_index_data.index = child_index
            child_node_exploration_index_data.max_path_value = child_node_max_path_value
            child_node_exploration_index_data.min_path_value = child_node_min_path_value
        else:
            assert child_node_exploration_index_data.max_path_value is not None
            assert child_node_exploration_index_data.min_path_value is not None
            child_node_exploration_index_data.index = min(
                child_node_exploration_index_data.index, child_index
            )
            child_node_exploration_index_data.max_path_value = min(
                child_node_max_path_value,
                child_node_exploration_index_data.max_path_value,
            )
            child_node_exploration_index_data.min_path_value = max(
                child_node_min_path_value,
                child_node_exploration_index_data.min_path_value,
            )


class UpdateIndexZipfFactoredProba:
    """
    A class that updates the exploration index of nodes in a tree using the Zipf factored probability strategy.
    """

    def update_root_node_index(
        self,
        root_node: AlgorithmNode,
        root_node_exploration_index_data: RecurZipfQuoolExplorationData | None,
    ) -> None:
        """
        Updates the exploration index of the root node in the tree using the Zipf factored probability strategy.

        Args:
            root_node (AlgorithmNode): The root node of the tree.
        """

        assert root_node_exploration_index_data is not None
        root_node_exploration_index_data.zipf_factored_proba = 1
        root_node_exploration_index_data.index = 0

    def update_node_indices(
        self,
        child_node: AlgorithmNode,
        parent_node: AlgorithmNode,
        parent_node_exploration_index_data: RecurZipfQuoolExplorationData| None,
        child_node_exploration_index_data: RecurZipfQuoolExplorationData|None,
        parent_node_content: None,
        tree: ValueTree,
        child_rank: int,
    ) -> None:
        """
        Updates the exploration index of a child node in the tree using the Zipf factored probability strategy.

        Args:
            child_node (AlgorithmNode): The child node to update.
            parent_node (AlgorithmNode): The parent node of the child node.
            tree (trees.MoveAndValueTree): The tree containing the nodes.
            child_rank (int): The rank of the child node among its siblings.
        """

        assert parent_node_exploration_index_data is not None

        parent_zipf_factored_proba: float | None = (
            parent_node_exploration_index_data.zipf_factored_proba
        )
        assert parent_zipf_factored_proba is not None

        child_zipf_proba: float = 1 / (child_rank + 1)
        child_zipf_factored_proba: float = child_zipf_proba * parent_zipf_factored_proba
        inverse_depth: float = 1 / (tree.node_depth(child_node) + 1)
        child_index: float = child_zipf_factored_proba * inverse_depth
        child_index = -child_index

        assert child_node_exploration_index_data is not None


        # the index of the child node is updated now
        # as a child node can have multiple parents we take the min if an index was previously computed
        if child_node_exploration_index_data.index is None:
            child_node_exploration_index_data.index = child_index
            child_node_exploration_index_data.zipf_factored_proba = (
                child_zipf_factored_proba
            )

        else:
            assert child_node_exploration_index_data.zipf_factored_proba is not None
            child_node_exploration_index_data.index = min(
                child_node_exploration_index_data.index, child_index
            )
            child_node_exploration_index_data.zipf_factored_proba = min(
                child_node_exploration_index_data.zipf_factored_proba,
                child_zipf_factored_proba,
            )


class UpdateIndexLocalMinChange:
    """
    A class that updates the exploration index of nodes in a tree using the local minimum change strategy.
    """

    def update_root_node_index(
        self,
        root_node: AlgorithmNode,
        root_node_exploration_index_data: IntervalExplo | None,
    ) -> None:
        """
        Updates the exploration index of the root node in the tree using the local minimum change strategy.

        Args:
            root_node (AlgorithmNode): The root node of the tree.
        """
        assert root_node_exploration_index_data is not None

        root_node_exploration_index_data.index = 0
        root_node_exploration_index_data.interval = Interval(
            min_value=-math.inf, max_value=math.inf
        )

    def update_node_indices(
        self,
        child_node: AlgorithmNode,
        parent_node: AlgorithmNode,
        parent_node_exploration_index_data: IntervalExplo | None,
        child_node_exploration_index_data: IntervalExplo | None,
        parent_node_content: HasTurn,
        tree: ValueTree,
        child_rank: int,
    ) -> None:
        """
        Updates the exploration index of a child node in the tree using the local minimum change strategy.

        Args:
            child_node (AlgorithmNode): The child node to update.
            parent_node (AlgorithmNode): The parent node of the child node.
            tree (trees.MoveAndValueTree): The tree containing the nodes.
            child_rank (int): The rank of the child node among its siblings.
        """

        assert parent_node_exploration_index_data is not None
        assert child_node_exploration_index_data is not None

        inter_level_interval: Interval | None = None
        local_index = None

        if parent_node_exploration_index_data.index is None:
            child_node_exploration_index_data.index = None
        else:
            assert parent_node_exploration_index_data.interval is not None
            if len(parent_node.tree_node.branches_children) == 1:
                local_index = parent_node_exploration_index_data.index
                inter_level_interval = parent_node_exploration_index_data.interval
            else:
                if parent_node_content.turn== Colors.WHITE:
                    best_move: BranchKey | None = (
                        parent_node.minmax_evaluation.best_move()
                    )
                    assert best_move is not None
                    best_child = parent_node.branches_children[best_move]
                    assert isinstance(best_child, AlgorithmNode)
                    second_best_move: BranchKey | None = (
                        parent_node.minmax_evaluation.second_best_move()
                    )
                    assert second_best_move is not None
                    second_best_child = parent_node.branches_children[second_best_move]
                    child_white_value = child_node.minmax_evaluation.get_value_white()
                    local_interval = Interval()
                    if child_node == best_child:
                        assert isinstance(second_best_child, AlgorithmNode)
                        local_interval.max_value = math.inf
                        local_interval.min_value = (
                            second_best_child.minmax_evaluation.get_value_white()
                        )
                    else:
                        local_interval.max_value = math.inf
                        local_interval.min_value = (
                            best_child.minmax_evaluation.get_value_white()
                        )

                    inter_level_interval = intersect_intervals(
                        local_interval, parent_node_exploration_index_data.interval
                    )
                    if inter_level_interval is not None:
                        local_index = distance_number_to_interval(
                            value=child_white_value, interval=inter_level_interval
                        )
                    else:
                        ...
                        local_index = None
                if parent_node_content.turn == Colors.BLACK:
                    best_move_black: BranchKey | None = (
                        parent_node.minmax_evaluation.best_move()
                    )
                    assert best_move_black is not None
                    best_child = parent_node.branches_children[best_move_black]
                    second_best_move_black: BranchKey | None = (
                        parent_node.minmax_evaluation.second_best_move()
                    )
                    assert second_best_move_black is not None
                    second_best_child = parent_node.branches_children[
                        second_best_move_black
                    ]
                    child_white_value = child_node.minmax_evaluation.get_value_white()
                    local_interval = Interval()
                    assert isinstance(best_child, AlgorithmNode)
                    if child_node == best_child:
                        assert isinstance(second_best_child, AlgorithmNode)
                        local_interval.max_value = (
                            second_best_child.minmax_evaluation.get_value_white()
                        )
                        local_interval.min_value = -math.inf
                    else:
                        local_interval.max_value = (
                            best_child.minmax_evaluation.get_value_white()
                        )
                        local_interval.min_value = -math.inf

                    inter_level_interval = intersect_intervals(
                        local_interval, parent_node_exploration_index_data.interval
                    )
                    if inter_level_interval is not None:
                        local_index = distance_number_to_interval(
                            value=child_white_value, interval=inter_level_interval
                        )
                    else:
                        local_index = None

            assert isinstance(child_node.exploration_index_data, IntervalExplo)

            if child_node.exploration_index_data.index is None:
                child_node.exploration_index_data.index = local_index
                child_node.exploration_index_data.interval = inter_level_interval

            elif local_index is not None:
                if local_index < child_node.exploration_index_data.index:
                    child_node.exploration_index_data.interval = inter_level_interval
                child_node.exploration_index_data.index = min(
                    child_node.exploration_index_data.index, local_index
                )


# TODO their might be ways to optimize the computation such as not recomptuing for the whole tree
def update_all_indices(
    tree: ValueTree, index_manager: NodeExplorationIndexManager
) -> None:
    """
    The idea is to compute an index $ind(n)$ for a node $n$ that measures the minimum amount of change
     in the value of all the nodes such that this node $n$ becomes the best.

    This can be computed recursively as :
    ind(n) = max( ind(parent(n),.5*abs(value(n)-value(parent(n))))

    Args:
        index_manager:
        tree: a tree

    Returns:

    """
    if isinstance(index_manager, NullNodeExplorationIndexManager):
        return

    tree_nodes: RangedDescendants = tree.descendants

    index_manager.update_root_node_index(
        root_node=tree.root_node,
        root_node_exploration_index_data=tree.root_node.exploration_index_data,
    )

    half_move: int
    for half_move in tree_nodes:
        # todo how are we sure that the hm comes in order?
        parent_node: ITreeNode
        for parent_node in tree_nodes[half_move].values():
            assert isinstance(parent_node, AlgorithmNode)
            child_node: ITreeNode | None
            # for child_node in parent_node.moves_children.values():
            move_rank: int
            move: BranchKey
            for move_rank, move in enumerate(
                parent_node.minmax_evaluation.moves_sorted_by_value_
            ):
                child_node = parent_node.branches_children[move]
                assert isinstance(child_node, AlgorithmNode)
                index_manager.update_node_indices(
                    child_node=child_node,
                    tree=tree,
                    child_rank=move_rank,
                    parent_node=parent_node,
                    parent_node_exploration_index_data=parent_node.exploration_index_data,
                    child_node_exploration_index_data=child_node.exploration_index_data,
                    parent_node_content=parent_node.tree_node.content
                )


# TODO their might be ways to optimize the computation such as not recomptuing for the whole tree


def print_all_indices(
    tree: ValueTree,
) -> None:
    """
    Prints the exploration indices of all nodes in the given tree.

    Args:
        tree (trees.MoveAndValueTree): The tree containing the nodes.

    Returns:
        None
    """
    tree_nodes: RangedDescendants = tree.descendants

    half_move: int
    for half_move in tree_nodes:
        # todo how are we sure that the hm comes in order?
        parent_node: ITreeNode
        for parent_node in tree_nodes[half_move].values():
            assert isinstance(parent_node, AlgorithmNode)
            if parent_node.exploration_index_data is not None:
                print(
                    "parent_node",
                    parent_node.tree_node.id,
                    parent_node.exploration_index_data.index,
                )
