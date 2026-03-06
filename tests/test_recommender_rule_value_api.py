from dataclasses import dataclass
from types import SimpleNamespace

from valanga import Color

from anemone.recommender_rule.recommender_rule import SoftmaxRule


@dataclass
class _ChildEval:
    score: float

    def get_score(self) -> float:
        return self.score


@dataclass
class _ChildNode:
    tree_evaluation: _ChildEval


@dataclass
class _RootNode:
    state: object
    branches_children: dict[int, _ChildNode | None]


def test_softmax_rule_uses_score_not_legacy_float_api() -> None:
    rule = SoftmaxRule(type="softmax", temperature=1.0)
    root = _RootNode(
        state=SimpleNamespace(turn=Color.WHITE),
        branches_children={
            0: _ChildNode(tree_evaluation=_ChildEval(score=0.1)),
            1: _ChildNode(tree_evaluation=_ChildEval(score=0.8)),
        },
    )

    policy = rule.policy(root)

    assert set(policy.probs) == {0, 1}
    assert policy.probs[1] > policy.probs[0]
