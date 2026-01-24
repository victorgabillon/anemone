"""
This module defines recommender rules for selecting branches in a tree-based selector.

The recommender rules are implemented as data classes that define a `__call__` method. The `__call__` method takes a
root node and a random generator, and returns a recommended branch.

The available recommender rule types are defined in the `RecommenderRuleTypes` enum.

The module also defines a `RecommenderRule` protocol that all recommender rule classes must implement.

Example usage:
    rule = AlmostEqualLogistic(type=RecommenderRuleTypes.AlmostEqualLogistic, temperature=0.5)
    branch = rule(tree, random_generator)
"""

from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Literal, Protocol

from valanga import BranchKey, State
from valanga.policy import BranchPolicy

from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)
from anemone.utils.small_tools import softmax


def sample_from_policy(policy: BranchPolicy, rng: Random) -> BranchKey:
    """Sample a branch key from a probability policy using a RNG."""
    branches = list(policy.probs.keys())
    weights = list(policy.probs.values())
    return rng.choices(branches, weights=weights, k=1)[0]


class RecommenderRule(Protocol):
    """
    Protocol for recommender rules.
    """

    type: str

    def policy[StateT: State](self, root_node: AlgorithmNode[StateT]) -> BranchPolicy:
        """Return the policy distribution for the root node."""
        ...

    def sample(self, policy: BranchPolicy, rng: Random) -> BranchKey:
        """Sample a branch key using the provided RNG."""
        ...


class RecommenderRuleTypes(str, Enum):
    """
    Enum class that defines the available recommender rule types.
    """

    ALMOST_EQUAL_LOGISTIC = "almost_equal_logistic"
    SOFTMAX = "softmax"


# theses are functions but i still use dataclasses instead
# of partial to be able to easily construct from yaml files using dacite


@dataclass(slots=True)
class AlmostEqualLogistic:
    """
    Almost Equal Logistic recommender rule that selects branches with nearly equal evaluations.
    """

    type: Literal["almost_equal_logistic"]
    temperature: float  # kept for config compatibility; rule uses minmax method

    def policy[StateT: State](self, root_node: AlgorithmNode[StateT]) -> BranchPolicy:
        """Compute a policy based on near-equal best branches."""
        best: list[BranchKey] = root_node.tree_evaluation.get_all_of_the_best_branches(
            how_equal="almost_equal_logistic"
        )

        # Fallback: if empty, uniform over all existing children
        if not best:
            best = [
                bk for bk, ch in root_node.branches_children.items() if ch is not None
            ]

        # If still empty, something is wrong (no legal branches / not expanded)
        if not best:
            return BranchPolicy(probs={})

        p = 1.0 / len(best)
        return BranchPolicy(probs={bk: p for bk in best})

    def sample(self, policy: BranchPolicy, rng: Random) -> BranchKey:
        """Sample a branch from the policy using the provided RNG."""
        return sample_from_policy(policy, rng)


@dataclass(slots=True)
class SoftmaxRule:
    """
    Softmax recommender rule that computes a softmax distribution over child evaluations.
    """

    type: Literal["softmax"]
    temperature: float

    def policy[StateT: State](self, root_node: AlgorithmNode[StateT]) -> BranchPolicy:
        """Compute a softmax policy over child evaluations."""
        branches: list[BranchKey] = []
        scores: list[float] = []

        for bk, child in root_node.branches_children.items():
            if child is None:
                continue
            branches.append(bk)
            score = root_node.tree_evaluation.subjective_value_of(child.tree_evaluation)
            scores.append(float(score))

        if not branches:
            return BranchPolicy(probs={})

        probs_list = softmax(scores, self.temperature)  # list[float] or Sequence[float]
        probs = {bk: float(p) for bk, p in zip(branches, probs_list, strict=True)}
        return BranchPolicy(probs=probs)

    def sample(self, policy: BranchPolicy, rng: Random) -> BranchKey:
        """Sample a branch from the policy using the provided RNG."""
        return sample_from_policy(policy, rng)


AllRecommendFunctionsArgs = AlmostEqualLogistic | SoftmaxRule
