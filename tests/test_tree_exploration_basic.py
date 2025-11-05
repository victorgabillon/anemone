"""
Basic test for tree exploration functionality.

This test creates minimal mock implementations of the chipiron dependencies
to allow testing the tree exploration functionality in a self-contained way.
"""

import random
from dataclasses import dataclass
from typing import Iterator

from valanga import BranchKey, BranchKeyGeneratorP

from anemone.node_factory import Base
from anemone.nodes.algorithm_node import AlgorithmNode
from anemone.nodes.tree_node import TreeNode
from anemone.trees.descendants import RangedDescendants
from anemone.trees.value_tree import ValueTree


@dataclass
class SimpleTreeExploration:
    """
    Simple TreeExploration implementation for testing.

    This mirrors the structure of the actual TreeExploration class
    but uses simple mock implementations for dependencies.
    """

    def print_info_during_move_computation(
        self, random_generator: random.Random
    ) -> None:
        """Print information during move computation."""
        current_best_move = "e2e4"  # Default move
        if self.tree.root_node.minmax_evaluation.best_move_sequence:
            current_best_move = str(
                self.tree.root_node.minmax_evaluation.best_move_sequence[0]
            )

        if random_generator.random() < 0.5:  # More frequent than original for testing
            print(f"FEN: {self.tree.root_node.board.fen}")
            str_progress = self.stopping_criterion.get_string_of_progress(self.tree)
            print(
                f"{str_progress} | current best move: {current_best_move} | current white value: {self.tree.root_node.minmax_evaluation.value_white_minmax}"
            )
            self.tree.root_node.minmax_evaluation.print_moves_sorted_by_value_and_exploration()
            self.tree_manager.print_best_line(tree=self.tree)


@dataclass
class SimpleBranchKeyGenerator:
    """
    Simple BranchKeyGenerator for testing.

    This mock implementation generates simple string keys.
    """

    all_generated_keys = [0, 1]

    # whether to sort the branch keys by their respective uci for easy comparison of various implementations
    sort_branch_keys = True

    def __iter__(self) -> Iterator[BranchKey]:
        """Returns an iterator over the branch keys."""
        return iter(self.all_generated_keys)

    def __next__(self) -> BranchKey:
        """Returns the next branch key."""
        if not hasattr(self, "_iterator"):
            self._iterator = iter(self.all_generated_keys)
        return next(self._iterator)

    def more_than_one_branch(self) -> bool:
        return True

    def get_all(self) -> list[BranchKey]:
        """Returns a list of all branch keys."""
        return [0, 1]

    def copy_with_reset(self) -> "SimpleBranchKeyGenerator":
        """Creates a copy of the legal move generator with an optional reset of generated moves.

        Returns:
            SimpleBranchKeyGenerator: A new instance of the legal move generator with the specified generated moves.
        """
        return SimpleBranchKeyGenerator()


@dataclass
class SimpleState:
    tag_: int

    """Simple state class for testing."""

    def tag(self) -> StateTag:
        """Returns the tag of the state.

        Returns:
            StateTag: The tag of the state.
        """
        return self.tag_

    @property
    def branch_keys(self) -> BranchKeyGeneratorP:
        """Returns the branch keys of the state.

        Returns:
            BranchKeyGeneratorP: The branch keys of the state.
        """
        return SimpleBranchKeyGenerator()


def test_tree_exploration_basic():
    """
    Basic test for TreeExploration.

    This test creates a TreeExploration instance and calls the explore function
    to demonstrate basic functionality.
    """
    print("=== Starting Basic Tree Exploration Test ===")

    def notify_progress(percent: int):
        print(f"Progress: {percent}%")

    state = SimpleState(tag_=1)

    base_tree_node_factory = Base[TreeNode]()

    tree_node: TreeNode = base_tree_node_factory.create(
        count=0,
        tree_depth=0,
        state=state,
        parent_node=None,
        branch_from_parent=None,
    )

    root_node = AlgorithmNode(
        tree_node=tree_node,
        minmax_evaluation=None,  # Replace with actual evaluation if needed
        exploration_index_data=None,
        state_representation=None,
    )

    value_tree: ValueTree = ValueTree(
        root_node=root_node, descendants=RangedDescendants()
    )

    # Create TreeExploration instance
    tree_exploration = SimpleTreeExploration(
        tree=value_tree,
        tree_manager=tree_manager,
        node_selector=node_selector,
        recommend_move_after_exploration=recommend_move_after_exploration,
        stopping_criterion=stopping_criterion,
        notify_percent_function=notify_progress,
    )

    print("✓ TreeExploration instance created successfully")

    # Test the explore function
    random_generator = random.Random(42)  # Use fixed seed for reproducible results

    print("✓ Starting tree exploration...")
    result = tree_exploration.explore(random_generator)

    print("✓ Exploration completed!")
    print(f"✓ Recommended move: {result.move_recommendation.move}")
    print(f"✓ Evaluation: {result.move_recommendation.evaluation}")

    # Basic assertions
    assert isinstance(result, SimpleTreeExplorationResult)
    assert hasattr(result, "move_recommendation")
    assert hasattr(result, "tree")
    assert result.move_recommendation.move is not None
    assert result.move_recommendation.move == "e2e4"  # Expected default move

    print("✓ All assertions passed!")
    print("=== Test Completed Successfully! ===")


if __name__ == "__main__":
    test_tree_exploration_basic()
