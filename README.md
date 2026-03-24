# anemone

`anemone` is a Python library for tree search over `valanga` game states. It
separates structural tree storage, runtime search state, node evaluation,
exploration indices, and selector/opening logic so different search policies can
share one coherent runtime.

## Overview

The main architectural split is:

- structure: `TreeNode` and `ITreeNode`
- runtime: `AlgorithmNode` and `TreeExploration`
- evaluation: direct evaluation plus backed-up tree evaluation
- exploration: per-node exploration-index payloads and tree-wide recomputation
- selection: selectors and opening instructions that decide what to expand next

## Installation

```bash
pip install anemone
```

Optional extras:

```bash
pip install anemone[nn]
pip install anemone[debug]
```

Debug snapshot export and the debug browser's SVG rendering also require the
system Graphviz executable (`dot`).

## Getting started

The canonical public entry point is `anemone.factory.create_search`.

- Use `anemone.factory` to build a runnable search runtime.
- Treat `anemone._runtime_assembly` and `anemone.search_factory` as internal
  wiring layers behind that API.
- Use `TreeAndValueBranchSelector` only when you want the higher-level
  one-shot recommendation wrapper instead of driving the runtime directly.

```python
from random import Random

from anemone.factory import SearchArgs, create_search

# Domain-specific collaborators omitted for brevity:
# - YourStateType
# - your_dynamics
# - starting_state
# - your_evaluator
# - your_selector_args
# - your_opening_type
# - your_stopping_criterion
# - your_recommender_rule
args = SearchArgs(
    node_selector=your_selector_args,
    opening_type=your_opening_type,
    stopping_criterion=your_stopping_criterion,
    recommender_rule=your_recommender_rule,
)

runtime = create_search(
    state_type=YourStateType,
    dynamics=your_dynamics,
    starting_state=starting_state,
    args=args,
    random_generator=Random(0),
    master_state_value_evaluator=your_evaluator,
    state_representation_factory=None,
)

result = runtime.explore(Random(0))
print(result.branch_recommendation.recommended_name)
```

## Core concepts

- `TreeNode`: the concrete structural node storing state, depth, and links.
- `AlgorithmNode`: the runtime wrapper that adds tree evaluation and
  exploration-index payloads.
- `direct_value`: the immediate evaluator output for one node.
- `backed_up_value`: the subtree-derived value propagated from children.
- exploration index: optional per-node search-priority data recomputed across
  the tree by a configured strategy.

## How it works

- `create_search(...)` assembles the runtime, direct evaluator, selector
  factory, tree-evaluation factory, and tree manager.
- The initial tree is created with a directly evaluated root node.
- A selector chooses which node or branch to open next.
- The tree manager expands the structure and creates any new runtime nodes.
- Direct evaluation fills `direct_value` on newly created nodes.
- Upward propagation recomputes `backed_up_value`, branch ordering, and
  principal variation.
- Exploration-index payloads are refreshed for the next iteration.

## Documentation

- [Search iteration architecture](docs/source/search_iteration_architecture.rst)
- [Node evaluation README](src/anemone/node_evaluation/README.md)
- [Exploration indices README](src/anemone/indices/README.md)
- [Node selector README](src/anemone/node_selector/README.md)
- [Nodes README](src/anemone/nodes/README.md)
- [Search factory README](src/anemone/search_factory/README.md)
