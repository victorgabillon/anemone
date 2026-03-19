"""Node evaluation utilities and evaluation families."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .common import canonical_value
    from .common.branch_frontier import BranchFrontierAware, BranchFrontierState
    from .common.branch_ordering import DecisionOrderedEvaluation
    from .common.canonical_value import ValueSemanticsError
    from .common.node_value_evaluation import NodeValueEvaluation
    from .common.principal_variation import PrincipalVariationState
    from .direct import (
        DirectValueInvariantError,
        EvaluationQueries,
        MasterStateValueEvaluator,
        NodeDirectEvaluator,
        NodeEvaluatorTypes,
        OverEventDetector,
        create_node_evaluator,
    )
    from .tree.adversarial.node_adversarial_evaluation import NodeAdversarialEvaluation
    from .tree.adversarial.node_minmax_evaluation import NodeMinmaxEvaluation
    from .tree.factory import NodeTreeEvaluationFactory, NodeTreeMinmaxEvaluationFactory
    from .tree.node_tree_evaluation import (
        BestBranchEquivalenceMode,
        NodeTreeEvaluation,
        NodeTreeEvaluationState,
    )
    from .tree.single_agent.factory import NodeMaxEvaluationFactory
    from .tree.single_agent.node_max_evaluation import NodeMaxEvaluation
    from .tree.single_agent.node_single_agent_evaluation import (
        NodeSingleAgentEvaluation,
    )

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "DecisionOrderedEvaluation": (
        ".common.branch_ordering",
        "DecisionOrderedEvaluation",
    ),
    "BranchFrontierAware": (".common.branch_frontier", "BranchFrontierAware"),
    "BranchFrontierState": (".common.branch_frontier", "BranchFrontierState"),
    "BestBranchEquivalenceMode": (
        ".tree.node_tree_evaluation",
        "BestBranchEquivalenceMode",
    ),
    "PrincipalVariationState": (
        ".common.principal_variation",
        "PrincipalVariationState",
    ),
    "DirectValueInvariantError": (".direct", "DirectValueInvariantError"),
    "EvaluationQueries": (".direct", "EvaluationQueries"),
    "MasterStateValueEvaluator": (".direct", "MasterStateValueEvaluator"),
    "NodeAdversarialEvaluation": (
        ".tree.adversarial.node_adversarial_evaluation",
        "NodeAdversarialEvaluation",
    ),
    "NodeDirectEvaluator": (".direct", "NodeDirectEvaluator"),
    "NodeEvaluatorTypes": (".direct", "NodeEvaluatorTypes"),
    "NodeMaxEvaluation": (
        ".tree.single_agent.node_max_evaluation",
        "NodeMaxEvaluation",
    ),
    "NodeMaxEvaluationFactory": (
        ".tree.single_agent.factory",
        "NodeMaxEvaluationFactory",
    ),
    "NodeMinmaxEvaluation": (
        ".tree.adversarial.node_minmax_evaluation",
        "NodeMinmaxEvaluation",
    ),
    "NodeSingleAgentEvaluation": (
        ".tree.single_agent.node_single_agent_evaluation",
        "NodeSingleAgentEvaluation",
    ),
    "NodeTreeEvaluationFactory": (".tree.factory", "NodeTreeEvaluationFactory"),
    "NodeTreeEvaluation": (".tree.node_tree_evaluation", "NodeTreeEvaluation"),
    "NodeTreeEvaluationState": (
        ".tree.node_tree_evaluation",
        "NodeTreeEvaluationState",
    ),
    "NodeTreeMinmaxEvaluationFactory": (
        ".tree.factory",
        "NodeTreeMinmaxEvaluationFactory",
    ),
    "NodeValueEvaluation": (".common.node_value_evaluation", "NodeValueEvaluation"),
    "OverEventDetector": (".direct", "OverEventDetector"),
    "ValueSemanticsError": (".common.canonical_value", "ValueSemanticsError"),
    "canonical_value": (".common.canonical_value", None),
    "create_node_evaluator": (".direct", "create_node_evaluator"),
}

__all__ = [
    "BranchFrontierAware",
    "BranchFrontierState",
    "BestBranchEquivalenceMode",
    "DecisionOrderedEvaluation",
    "DirectValueInvariantError",
    "EvaluationQueries",
    "MasterStateValueEvaluator",
    "NodeAdversarialEvaluation",
    "NodeDirectEvaluator",
    "NodeEvaluatorTypes",
    "NodeMaxEvaluation",
    "NodeMaxEvaluationFactory",
    "NodeMinmaxEvaluation",
    "NodeSingleAgentEvaluation",
    "NodeTreeEvaluation",
    "NodeTreeEvaluationFactory",
    "NodeTreeEvaluationState",
    "NodeTreeMinmaxEvaluationFactory",
    "NodeValueEvaluation",
    "OverEventDetector",
    "PrincipalVariationState",
    "ValueSemanticsError",
    "canonical_value",
    "create_node_evaluator",
]


def __getattr__(name: str) -> Any:
    """Lazily expose the public node-evaluation API without extra import-time work."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    module = import_module(module_name, __name__)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the public lazy exports for interactive discovery."""
    return sorted((*globals().keys(), *__all__))
