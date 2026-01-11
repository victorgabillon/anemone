# anemone

`anemone` is a Python library for tree search over `valanga` game states. It builds a
shared tree graph and layers algorithm-specific wrappers on top so you can plug in
node evaluation, exploration indices, and selection policies for "tree and value"
searches.

## Highlights

- Tree-and-value exploration pipeline driven by `TreeAndValueBranchSelector`.
- Modular factories for node evaluation, selection, index computation, and tree
  management.
- Pluggable stopping criteria and recommender rules for final branch selection.
- Optional torch-based evaluator for batched neural evaluations.

## Installation

```bash
pip install anemone
```

Optional torch integration:

```bash
pip install anemone[nn]
```

## Quick start

`anemone` exposes factory helpers to build a branch selector configured with your
node selector, evaluation, and stopping-criterion choices. At runtime you feed it a
`valanga` state and a seed to get back a branch recommendation.

```python
from random import Random

from anemone import TreeAndValuePlayerArgs, create_tree_and_value_branch_selector
from anemone.node_selector.factory import UniformArgs
from anemone.node_selector.node_selector_types import NodeSelectorType
from anemone.progress_monitor.progress_monitor import (
    StoppingCriterionTypes,
    TreeMoveLimitArgs,
)
from anemone.recommender_rule.recommender_rule import SoftmaxRule

# Populate the pieces specific to your game domain.
args = TreeAndValuePlayerArgs(
    node_selector=UniformArgs(type=NodeSelectorType.UNIFORM),
    opening_type=None,
    stopping_criterion=TreeMoveLimitArgs(
        type=StoppingCriterionTypes.TREE_MOVE_LIMIT,
        tree_move_limit=100,
    ),
    recommender_rule=SoftmaxRule(type="softmax", temperature=1.0),
)

selector = create_tree_and_value_branch_selector(
    state_type=YourStateType,
    args=args,
    random_generator=Random(0),
    master_state_evaluator=your_state_evaluator,
    state_representation_factory=None,
    queue_progress_player=None,
)

recommendation = selector.select_branch(state=current_state, selection_seed=0)
print(recommendation.branch_key)
```

## Design

This codebase follows a “core node + wrappers” pattern.

- **`TreeNode` (core)**
  - `TreeNode` is the canonical, shared data structure.
  - It stores the graph structure: `branches_children` and `parent_nodes`.
  - There is conceptually a single tree/graph of `TreeNode`s.

- **Wrappers implement `ITreeNode`**
  - Higher-level nodes (e.g. `AlgorithmNode`) wrap a `TreeNode` and add algorithm-specific state:
    evaluation, indices, representations, etc.
  - Wrappers expose navigation by delegating to the underlying `TreeNode`.

- **Homogeneity at the wrapper level**
  - Even though `TreeNode` is the core place where connections are stored, each wrapper is intended to be
    *closed under parent/child links*:
    - a wrapper’s `branches_children` and `parent_nodes` contain that same wrapper type.
    - today this is typically either “all `TreeNode`” or “all `AlgorithmNode`”.
    - in the future, another wrapper can exist (still implementing `ITreeNode`), and it should also be
      homogeneous within itself.

The practical motivation is:
- algorithms can be written against `ITreeNode` (for navigation) and against wrappers like `AlgorithmNode`
  (for algorithm-specific fields),
- while keeping a single shared underlying structure that can be accessed consistently from any wrapper.

## Repository layout

Each important package folder includes a local README with details. Start with:

- `src/anemone/` for the main search pipeline and public entry points.
- `src/anemone/node_selector/` for selection strategies (Uniform, RecurZipf, Sequool).
- `src/anemone/node_evaluation/` for direct evaluation and minmax tree evaluation.
- `src/anemone/tree_manager/`, `src/anemone/trees/`, and `src/anemone/updates/` for tree construction,
  expansion, and backpropagation.
- `src/anemone/indices/` for exploration index computation and updates.
- `tests/` for index and tree-building fixtures.
