"""
This module contains functions for testing the indices used in the branch selector tree.

The main functions in this module are:
- `make_tree_from_file`: Creates a branch and value tree from a YAML file.
- `check_from_file`: Compares the indices computed from the tree with the expected indices from a YAML file.
- `check_index`: Tests the indices for a specific index computation type and tree file.
- `test_indices`: Runs the index tests for multiple index computation types and tree files.
"""

from enum import Enum
from math import isclose
from random import Random
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from valanga import Color

import anemone.node_factory as node_factory
import anemone.search_factory as search_factories
import anemone.trees as trees
from anemone.indices.index_manager.factory import (
    create_exploration_index_manager,
)
from anemone.indices.index_manager.node_exploration_manager import (
    update_all_indices,
)
from anemone.indices.node_indices.index_types import (
    IndexComputationType,
)
from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    NodeDirectEvaluator,
)
from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation_factory import (
    NodeTreeMinmaxEvaluationFactory,
)
from anemone.node_selector.node_selector import NodeSelector
from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
    create_instructions_to_open_all_branches,
)
from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from anemone.progress_monitor.progress_monitor import ProgressMonitor
from anemone.tree_manager.factory import create_algorithm_node_tree_manager
from anemone.tree_manager.tree_expander import (
    TreeExpansion,
    TreeExpansions,
)
from anemone.trees.factory import ValueTreeFactory
from anemone.trees.tree import (
    Tree,
)
from anemone.utils.small_tools import path
from tests.fake_yaml_game import (
    FakeYamlState,
    MasterStateEvaluatorFromYaml,
    build_yaml_maps,
)

if TYPE_CHECKING:
    from anemone.nodes.itree_node import ITreeNode


class StopWhenTreeHasAllYamlNodes(ProgressMonitor[AlgorithmNode[FakeYamlState]]):
    def __init__(self, expected_node_count: int):
        self.expected_node_count = expected_node_count

    def should_we_continue(self, tree) -> bool:
        n = sum(len(tree.descendants[d]) for d in tree.descendants)
        return n < self.expected_node_count

    def respectful_opening_instructions(self, opening_instructions, tree):
        return opening_instructions

    def get_string_of_progress(self, tree) -> str:
        n = sum(len(tree.descendants[d]) for d in tree.descendants)
        return f"{n}/{self.expected_node_count} nodes"

    def notify_percent_progress(self, tree, notify_percent_function):
        return


def _pick_node_to_fully_open_bfs(tree: Tree[AlgorithmNode]) -> AlgorithmNode | None:
    for depth in tree.descendants:
        for node in tree.descendants[depth].values():
            # if node has at least one child missing, open it fully
            branches = node.state.branch_keys.get_all()
            if not branches:
                continue
            for bk in branches:
                if node.branches_children.get(bk) is None:
                    return node
    return None


def _instructions_open_all_children(
    node: AlgorithmNode,
) -> OpeningInstructions[AlgorithmNode]:
    # mimic OpeningInstructor.ALL_CHILDREN
    node.tree_node.all_branches_generated = True
    branches = node.state.branch_keys.get_all()
    return create_instructions_to_open_all_branches(
        branches_to_play=branches, node_to_open=node
    )


class OpenAllInBfsOrder(NodeSelector[AlgorithmNode[FakeYamlState]]):
    def choose_node_and_branch_to_open(self, tree, latest_tree_expansions):
        # BFS by depth
        for depth in tree.descendants:
            for node in tree.descendants[depth].values():
                # all branches for this node (0..n-1)
                for bk in node.state.branch_keys.get_all():
                    # not opened yet?
                    child = node.branches_children.get(bk, None)
                    if child is None:
                        return OpeningInstructions(
                            {node.id: OpeningInstruction(node_to_open=node, branch=bk)}
                        )

        return OpeningInstructions({})


class TestResult(Enum):
    """
    Enumeration for the test results.
    """

    __test__ = False
    PASSED = 0
    FAILED = 1
    WARNING = 2


def build_tree_from_yaml_clean(
    file_path: str, index_computation: IndexComputationType
) -> Tree[AlgorithmNode[FakeYamlState]]:
    yaml_nodes = yaml.safe_load(open(file_path))["nodes"]
    children_by_id, value_by_id = build_yaml_maps(yaml_nodes)
    expected_nodes = len(value_by_id)

    master_eval = MasterStateEvaluatorFromYaml(value_by_id=value_by_id)
    node_direct_eval = NodeDirectEvaluator(master_state_evaluator=master_eval)

    # factories like prod
    tree_node_factory = node_factory.TreeNodeFactory[Any]()
    search_factory = search_factories.SearchFactory(
        node_selector_args=None,
        opening_type=None,
        random_generator=Random(0),
        index_computation=index_computation,
    )

    algo_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=tree_node_factory,
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeTreeMinmaxEvaluationFactory(),
        exploration_index_data_create=search_factory.node_index_create,
    )

    tree_factory = ValueTreeFactory(
        node_factory=algo_factory,
        node_direct_evaluator=node_direct_eval,
    )

    tree_manager = create_algorithm_node_tree_manager(
        algorithm_node_factory=algo_factory,
        node_direct_evaluator=node_direct_eval,
        index_computation=index_computation,
        index_updater=search_factory.create_node_index_updater(),
    )

    root_state = FakeYamlState(
        node_id=0, children_by_id=children_by_id, turn=Color.WHITE
    )
    tree = tree_factory.create(starting_state=root_state)

    # simple exploration loop (no recommender needed)
    expansions = TreeExpansions()
    expansions.add_creation(
        TreeExpansion(
            child_node=tree.root_node,
            parent_node=None,
            state_modifications=None,
            creation_child_node=True,
            branch_key=None,
        )
    )
    while tree.nodes_count < expected_nodes:
        node = _pick_node_to_fully_open_bfs(tree)
        if node is None:
            break

        instr = _instructions_open_all_children(node)

        expansions = tree_manager.open_instructions(
            tree=tree, opening_instructions=instr
        )
        tree_manager.update_backward(tree_expansions=expansions)

    return tree


def check_from_file(file_path: path, tree: Tree[AlgorithmNode]) -> None:
    """
    Check the values in the given file against the values in the tree.

    Args:
        file_path (str): The path to the file containing the values to check.
        tree (ValueTree): The tree containing the values to compare against.

    Returns:
        None
    """
    with open(file_path) as file:
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
                    yaml_index, parent_node.exploration_index_data.index, abs_tol=1e-6
                )


def check_index(index_computation: IndexComputationType, tree_file: path) -> TestResult:
    tree_path = f"tests/data/trees/{tree_file}/{tree_file}.yaml"

    tree = build_tree_from_yaml_clean(tree_path, index_computation)

    index_manager = create_exploration_index_manager(
        index_computation=index_computation
    )
    update_all_indices(tree, index_manager)

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
