# Node evaluation

Node evaluation is split into two layers:

- **Direct evaluation** (`node_direct_evaluation/`): evaluates a single game
  state and exposes `NodeDirectEvaluator` along with the `MasterStateEvaluator`
  protocol.
- **Tree evaluation** (`node_tree_evaluation/`): aggregates child evaluations and
  computes minmax values, best-branch sequences, and terminal-state tracking.

Factories in these modules are used by the main `factory.py` entry points to
build the evaluation pipeline for the tree-and-value search.
