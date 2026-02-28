"""Integration tests for backward propagation through tree manager updates."""

from math import isclose
from pathlib import Path
from typing import Any

import yaml
from valanga import Color

import anemone.node_factory as node_factory
from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    NodeDirectEvaluator,
)
from anemone.node_evaluation.node_tree_evaluation.node_tree_evaluation_factory import (
    NodeTreeMinmaxEvaluationFactory,
)
from anemone.node_selector.opening_instructions import (
    OpeningInstruction,
    OpeningInstructions,
)
from anemone.tree_manager.factory import create_algorithm_node_tree_manager
from anemone.trees.factory import ValueTreeFactory
from tests.fake_yaml_game import (
    FakeYamlDynamics,
    FakeYamlState,
    MasterStateEvaluatorFromYaml,
    build_yaml_maps,
)

TREE_PATH = Path(__file__).parent / "data/trees/value_backprop/tree_simple.yaml"


def _create_tree_and_manager() -> tuple[Any, Any]:
    with TREE_PATH.open("r", encoding="utf-8") as file:
        yaml_nodes = yaml.safe_load(file)["nodes"]
    children_by_id, value_by_id = build_yaml_maps(yaml_nodes)

    dynamics = FakeYamlDynamics()
    node_direct_eval = NodeDirectEvaluator(
        master_state_evaluator=MasterStateEvaluatorFromYaml(value_by_id=value_by_id)
    )

    algo_factory = node_factory.AlgorithmNodeFactory(
        tree_node_factory=node_factory.TreeNodeFactory[Any](),
        state_representation_factory=None,
        node_tree_evaluation_factory=NodeTreeMinmaxEvaluationFactory(),
        exploration_index_data_create=lambda _: None,
    )

    tree_factory = ValueTreeFactory(
        node_factory=algo_factory,
        node_direct_evaluator=node_direct_eval,
    )

    tree_manager = create_algorithm_node_tree_manager(
        algorithm_node_factory=algo_factory,
        node_direct_evaluator=node_direct_eval,
        dynamics=dynamics,
        index_computation=None,
        index_updater=None,
    )

    root_state = FakeYamlState(node_id=0, children_by_id=children_by_id, turn=Color.WHITE)
    tree = tree_factory.create(starting_state=root_state)
    return tree, tree_manager


def _open_branch(tree: Any, tree_manager: Any, node: Any, branch: int) -> None:
    instructions = OpeningInstructions(
        {
            node.tree_node.id: OpeningInstruction(node_to_open=node, branch=branch),
        }
    )
    expansions = tree_manager.open_instructions(tree=tree, opening_instructions=instructions)
    tree_manager.update_backward(tree_expansions=expansions)


def test_single_expansion_propagates_to_root() -> None:
    tree, tree_manager = _create_tree_and_manager()

    _open_branch(tree, tree_manager, tree.root_node, branch=0)

    assert isclose(tree.root_node.tree_evaluation.get_value_white(), 0.2)
    assert tree.root_node.tree_evaluation.best_branch_sequence[:1] == [0]


def test_second_expansion_can_change_root_choice() -> None:
    tree, tree_manager = _create_tree_and_manager()

    _open_branch(tree, tree_manager, tree.root_node, branch=0)
    _open_branch(tree, tree_manager, tree.root_node, branch=1)

    assert isclose(tree.root_node.tree_evaluation.get_value_white(), 0.6)
    assert tree.root_node.tree_evaluation.best_branch_sequence[:1] == [1]


def test_deep_leaf_update_propagates_through_intermediate_node() -> None:
    tree, tree_manager = _create_tree_and_manager()

    _open_branch(tree, tree_manager, tree.root_node, branch=0)
    node_a = tree.root_node.branches_children[0]
    assert node_a is not None

    _open_branch(tree, tree_manager, node_a, branch=0)

    assert isclose(node_a.tree_evaluation.get_value_white(), -0.8)
    assert node_a.tree_evaluation.best_branch_sequence[:1] == [0]

    assert isclose(tree.root_node.tree_evaluation.get_value_white(), -0.8)
    assert tree.root_node.tree_evaluation.best_branch_sequence[:2] == [0, 0]
