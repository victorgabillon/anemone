# Direct node evaluation

This folder contains the direct evaluation layer used to score a single state.

## Key pieces

- `node_direct_evaluator.py`: `NodeDirectEvaluator` and the
  `MasterStateEvaluator` protocol, plus batching helpers and evaluation queries.
- `factory.py`: Helper to create the evaluator from a
  `MasterStateEvaluator` implementation.

Direct evaluations feed into the tree-evaluation layer to seed minmax values for
leaf nodes.
