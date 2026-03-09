# Tree-based evaluation

Tree evaluation aggregates direct evaluations across children to compute the
minmax value of each node.

## Key pieces

- `node_adversarial_evaluation.py`: `NodeAdversarialEvaluation` protocol for
  adversarial backup, branch choice, and minimax semantics layered on canonical
  values.
- `node_tree_evaluation.py`: Transitional alias for the adversarial protocol.
- `node_minmax_evaluation.py`: Minmax implementation that tracks best branches,
  per-branch ordering, and terminal status.
- `node_tree_evaluation_factory.py`: Factories for constructing evaluation
  implementations (including the default minmax evaluator).
