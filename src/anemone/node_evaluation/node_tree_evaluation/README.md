# Tree-based evaluation

Tree evaluation aggregates direct evaluations across children to compute the
minmax value of each node.

## Key pieces

- `node_tree_evaluation.py`: `NodeTreeEvaluation` protocol defining the shared
  interface for tree evaluation.
- `node_minmax_evaluation.py`: Minmax implementation that tracks best branches,
  per-branch ordering, and terminal status.
- `node_tree_evaluation_factory.py`: Factories for constructing evaluation
  implementations (including the default minmax evaluator).
