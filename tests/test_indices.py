"""
This module contains functions for testing the indices used in the move selector tree.

The main functions in this module are:
- `make_tree_from_file`: Creates a move and value tree from a YAML file.
- `check_from_file`: Compares the indices computed from the tree with the expected indices from a YAML file.
- `check_index`: Tests the indices for a specific index computation type and tree file.
- `test_indices`: Runs the index tests for multiple index computation types and tree files.
"""

from calendar import c
from enum import Enum
from math import isclose
from typing import TYPE_CHECKING, Any

import chess

from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import MasterStateEvaluator, NodeDirectEvaluator, OverEventDetector
from anemone.node_selector.opening_instructions import OpeningInstruction, OpeningInstructions
import pytest
from valanga import HasTurn
import yaml
from atomheart import BoardChi, create_board_chi_from_pychess_board, ValangaChessState

from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation_factory import NodeTreeMinmaxEvaluationFactory
import anemone.node_factory as node_factory
import anemone.search_factory as search_factories
import anemone.tree_manager as tree_manager
import anemone.trees as trees
from anemone.indices.index_manager.factory import (
    create_exploration_index_manager,
)
from anemone.indices.index_manager.node_exploration_manager import (
    NodeExplorationIndexManager,
    update_all_indices,
)
from anemone.indices.node_indices.index_types import (
    IndexComputationType,
)
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from anemone.tree_manager.tree_expander import (
    TreeExpansion,
    TreeExpansions,
)
from anemone.trees.descendants import RangedDescendants
from anemone.trees.tree import (
    Tree,
)
from anemone.utils.small_tools import path

if TYPE_CHECKING:
    from anemone.basics import TreeDepth
    from anemone.nodes.itree_node import ITreeNode


class TestResult(Enum):
    """
    Enumeration for the test results.
    """

    __test__ = False
    PASSED = 0
    FAILED = 1
    WARNING = 2

class MasterStateEvaluatorFromFile(MasterStateEvaluator):
    """
    A MasterStateEvaluator that reads evaluations from a YAML file.
    """
    over: OverEventDetector


    yaml_nodes: list[dict[str, Any]]

    def __init__(self, yaml_nodes: list[dict[str, Any]]) -> None:
        self.yaml_nodes = yaml_nodes

    def value_white(self, state: State) -> float: ...
        sss

def make_tree_from_file(
    file_path: path, index_computation: IndexComputationType
) -> Tree[AlgorithmNode]:
    """
    Creates a move and value tree from a file.

    Args:
        file_path (path): The path to the file containing the tree data.
        index_computation (IndexComputationType): The type of index computation to use.

    Returns:
        ValueTree: The created move and value tree.
    """

    print(f"make_tree_from_file from {file_path}")

    # atm it is very ad hoc to test index so takes a lots of shortcut, will be made more general when needed
    with open(file_path, "r", encoding="utf-8") as file:
        tree_yaml = yaml.safe_load(file)
    yaml_nodes = tree_yaml["nodes"]



    tree_node_factory: node_factory.TreeNodeFactory[Any] = node_factory.TreeNodeFactory(
        
    )

    search_factory: search_factories.SearchFactoryP = search_factories.SearchFactory(
        node_selector_args=None,
        opening_type=None,
        random_generator=None,
        index_computation=index_computation,
    )

    algorithm_node_factory: node_factory.AlgorithmNodeFactory
    algorithm_node_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=tree_node_factory,
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeTreeMinmaxEvaluationFactory(),
        exploration_index_data_create=search_factory.node_index_create,
    )
    descendants: RangedDescendants = RangedDescendants()

    master_state_evaluator= MasterStateEvaluatorFromFile(yaml_nodes=yaml_nodes)

    node_direct_evaluator = NodeDirectEvaluator(
        master_state_evaluator=master_state_evaluator  )

    algo_tree_manager: tree_manager.AlgorithmNodeTreeManager = (
        tree_manager.create_algorithm_node_tree_manager(
            node_direct_evaluator=node_direct_evaluator,
            algorithm_node_factory=algorithm_node_factory,
            index_computation=index_computation,
            index_updater=None,
        )
    )

    tree_depths: dict[int, TreeDepth] = {}
    id_nodes: dict[int, AlgorithmNode] = {}
    move_and_value_tree: Tree[AlgorithmNode] | None = None

    root_node: ITreeNode | None = None


    tree_expansions : TreeExpansions[AlgorithmNode]

    for yaml_node in yaml_nodes:
        print("yaml_node", yaml_node)
        print("tree", move_and_value_tree.descendants if move_and_value_tree else None)
        if yaml_node["id"] == 0:
            tree_expansions = TreeExpansions()

            board = chess.Board.from_chess960_pos(yaml_node["id"])
            board.turn = chess.WHITE
            board_chi :BoardChi = create_board_chi_from_pychess_board(board)
            board_chi.legal_moves.get_all()
            board_state = ValangaChessState(board=board_chi)

            root_node = algorithm_node_factory.create(
                state=board_state,
                tree_depth=0,
                count=yaml_node["id"],
                parent_node=None,
                branch_from_parent=None,
                modifications=None,
            )
            assert isinstance(root_node, AlgorithmNode)
            root_node.tree_evaluation.value_white_minmax = yaml_node["value"]
            tree_depths[yaml_node["id"]] = 0
            id_nodes[yaml_node["id"]] = root_node

            descendants.add_descendant(root_node)
            move_and_value_tree = Tree(
                root_node=root_node, descendants=descendants
            )
            tree_expansions.add(
                TreeExpansion(
                    child_node=root_node,
                    parent_node=None,
                    state_modifications=None,
                    creation_child_node=True,
                    branch_key=None,
                )
            )
            root_node.tree_node.all_branches_generated = True
            # algo_tree_manager.update_backward(tree_expansions=tree_expansions)

        else:
            tree_expansions = TreeExpansions()
            first_parent = yaml_node["parents"]
            tree_depth = tree_depths[first_parent] + 1
            tree_depths[yaml_node["id"]] = tree_depth
            parent_node = id_nodes[first_parent]
            board = chess.Board.from_chess960_pos(yaml_node["id"])


            board.turn = not parent_node.tree_node.state.turn
            board_chi = create_board_chi_from_pychess_board(chess_board=board)
            board_state = ValangaChessState(board=board_chi)

            assert move_and_value_tree is not None

            
            tree_expansions=algo_tree_manager.open_instructions(
                tree=move_and_value_tree,
                opening_instructions=OpeningInstructions(
                    {
                        yaml_node["id"]: OpeningInstruction(
                            node_to_open=parent_node, branch=0
                        )
                    }
                ),
            )

            #tree_expansion: TreeExpansion[AlgorithmNode] = algo_tree_manager.tree_manager.open_tree_expansion_from_state(
            #    tree=move_and_value_tree,
            #    parent_node=parent_node,
            #    state=board_state,
            #    modifications=None,
            #    branch=yaml_node["id"],
            #)
            #tree_expansions.add(
            #    TreeExpansion(
            #        child_node=tree_expansion.child_node,
            #        parent_node=tree_expansion.parent_node,
            #        state_modifications=tree_expansion.state_modifications,
            #        creation_child_node=tree_expansion.creation_child_node,
            #        branch_key=yaml_node["id"],
            #    )
            #)
            tree_expansion: TreeExpansion[AlgorithmNode]|None = None
            for a in tree_expansions:
                print("expansion with creation", a)
                tree_expansion =a
            assert tree_expansion is not None
            assert isinstance(tree_expansion.child_node, AlgorithmNode)
            tree_expansion.child_node.tree_node.all_branches_generated = True
            id_nodes[yaml_node["id"]] = tree_expansion.child_node
            tree_expansion.child_node.tree_evaluation.value_white_minmax = yaml_node[
                "value"
            ]
            tree_expansion.child_node.tree_evaluation.value_white_direct_evaluation = (
                yaml_node["value"]
            )
            assert tree_expansion.branch_key is not None
            parent_node.tree_evaluation.branches_not_over.append(tree_expansion.branch_key)
            algo_tree_manager.update_backward(tree_expansions=tree_expansions)

    # print('move_and_value_tree', move_and_value_tree.descendants)
    assert move_and_value_tree is not None
    return move_and_value_tree


def check_from_file(file_path: path, tree: Tree[AlgorithmNode]) -> None:
    """
    Check the values in the given file against the values in the tree.

    Args:
        file_path (str): The path to the file containing the values to check.
        tree (ValueTree): The tree containing the values to compare against.

    Returns:
        None
    """
    with open(file_path, "r") as file:
        tree_yaml = yaml.safe_load(file)
    print("tree", tree_yaml)
    yaml_nodes = tree_yaml["nodes"]

    tree_nodes: trees.RangedDescendants = tree.descendants

    tree_depth: int
    for tree_depth in tree_nodes:
        # todo how are we sure that the hm comes in order?
        # print('hmv', tree_depth)
        parent_node: ITreeNode
        for parent_node in tree_nodes[tree_depth].values():
            assert isinstance(parent_node, AlgorithmNode)
            yaml_index = eval(str(yaml_nodes[parent_node.id]["index"]))
            assert parent_node.exploration_index_data is not None
            print(
                f"id {parent_node.id} expected value {yaml_index} "
                f"|| computed value {parent_node.exploration_index_data.index}"
                f" {type(yaml_nodes[parent_node.id]['index'])}"
                f" {type(parent_node.exploration_index_data.index)}"
                f"{(yaml_index == parent_node.exploration_index_data.index)}"
            )
            if yaml_index is None:
                assert parent_node.exploration_index_data.index is None
            else:
                assert parent_node.exploration_index_data.index is not None
                assert isclose(
                    yaml_index, parent_node.exploration_index_data.index, abs_tol=1e-8
                )


def check_index(index_computation: IndexComputationType, tree_file: path) -> TestResult:
    """
    Checks the index for a given tree file and index computation type.

    Args:
        index_computation (IndexComputationType): The type of index computation.
        tree_file (path): The path to the tree file.

    Returns:
        TestResult: The result of the index check.

    Raises:
        None

    """

    tree_path = f"tests/data/trees/{tree_file}/{tree_file}.yaml"
    tree: Tree = make_tree_from_file(
        index_computation=index_computation, file_path=tree_path
    )

    index_manager: NodeExplorationIndexManager = create_exploration_index_manager(
        index_computation=index_computation
    )

    print("index_manager", index_manager)
    update_all_indices(tree, index_manager)
    # print_all_indices(
    #    tree
    # )
    file_index = (
        f"tests/data/trees/{tree_file}/{tree_file}_{index_computation.value}.yaml"
    )
    check_from_file(file_path=file_index, tree=tree)

    return TestResult.PASSED

@pytest.mark.integration
def test_indices() -> None:
    """
    Test the index computations on multiple tree files.

    This function iterates over a list of index computations and tree files,
    and performs a test for each combination. The results of the tests are
    stored in a dictionary.

    Returns:
        None
    """
    index_computations: list[IndexComputationType] = [
        IndexComputationType.MIN_GLOBAL_CHANGE,
        IndexComputationType.RECUR_ZIPF,
        IndexComputationType.MIN_LOCAL_CHANGE,
    ]

    tree_files = ["tree_1", "tree_2"]

    results: dict[TestResult, int] = {}
    for tree_file in tree_files:
        if tree_file == "tree_2":
            index_computations_ = [
                IndexComputationType.MIN_GLOBAL_CHANGE,
                IndexComputationType.MIN_LOCAL_CHANGE,
            ]
        else:
            index_computations_ = index_computations

        for index_computation in index_computations_:
            print(f"---testing {index_computation} on {tree_file}")
            res: TestResult = check_index(
                index_computation=index_computation, tree_file=tree_file
            )
            if res in results:
                results[res] += 1
            else:
                results[res] = 1
    print(f"finished Test: {results}")
    assert results[TestResult.PASSED] == 5


if __name__ == "__main__":
    test_indices()
