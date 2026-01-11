# anemone package

This directory hosts the core package for the `anemone` tree-search library. The
public entry points are exposed via `anemone.__init__` and the factory helpers in
`factory.py`.

## Key modules

- `factory.py` and `tree_and_value_branch_selector.py`: Build and run the main
  tree-and-value branch selector.
- `tree_exploration.py`: Orchestrates a single search, including node selection,
  tree expansion, updates, and recommendation.
- `node_factory/`, `nodes/`: Create and represent the core `TreeNode` structure
  along with algorithm-specific wrappers.
- `node_evaluation/`: Direct evaluators plus minmax tree-evaluation logic.
- `node_selector/`: Node-opening strategies and opening instructions.
- `tree_manager/`, `trees/`, `updates/`: Tree construction, expansion, and
  backpropagation.
- `indices/`: Exploration index data and update workflows.
- `progress_monitor/`: Stopping criteria and progress reporting.
- `recommender_rule/`: Sampling rules for the final branch recommendation.
- `nn/`: Optional torch-backed evaluator integration.
- `utils/`: Shared helper utilities.
