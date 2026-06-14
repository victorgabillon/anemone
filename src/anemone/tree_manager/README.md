# Tree manager

Tree managers coordinate how nodes are opened, expanded, and updated.

Nodes can be unopened, partially opened, or fully opened. A node having children
does not imply that all legal branches are open. `all_branches_generated` means
no legal branch remains unopened, and opening all children opens only the
remaining legal branches.

## Key pieces

- `tree_manager.py`: Base `TreeManager` for opening nodes and tracking
  expansions.
- `branch_opening_service.py`: one-branch opening primitive that records
  `TreeExpansion` objects and runs per-branch callbacks such as branch-frontier
  bookkeeping.
- `opening_expansion_executor.py`: one-ply `OpeningInstructions` executor that
  opens requested branches and synchronizes touched parents once per batch.
- `algorithm_node_tree_manager.py`: `AlgorithmNodeTreeManager` implementation
  for `AlgorithmNode` trees.
- `tree_expander.py`: `TreeExpansion` and `TreeExpansions` helpers describing
  node creation events.
- `factory.py`: `create_algorithm_node_tree_manager` convenience constructor.

This split keeps `TreeManager` as the structural single-branch open/link
primitive while allowing future expansion executors, such as rollout expansion,
to reuse the same branch-opening path.

Deterministic materialized rollouts live in `anemone.rollouts`. A
`RolloutOpeningExpansionExecutor` can be injected where an
`OpeningExpansionExecutor` is accepted; it first opens the selected instruction
edge and can then continue through newly created child nodes by asking a rollout
action selector to choose among legal branches. Choosing an openable branch
materializes a new edge; choosing an already-opened branch traverses the
existing child link without recording a `TreeExpansion`. Existing-node
connections from newly materialized branches are recorded and stop rollout by
default. The default runtime remains one-ply unless a rollout executor is
explicitly configured.

Rollout expansion is selector-agnostic. It depends on `OpeningInstructions`,
`BranchOpeningService`, `SearchDynamics`, and opening-status helpers; it does
not depend on Linoo, Uniform, Sequool, RecurZipf, or selector-specific state.
Node selectors decide which not-fully-opened nodes and branches to open.
Expansion executors decide how much tree to materialize after those initial
openings. Rollout is therefore an expansion materialization strategy, not a
node-selection algorithm.

Opening expansion can be configured independently from node selection:

```python
OpeningExpansionConfig(
    kind=OpeningExpansionKind.ROLLOUT,
    rollout=RolloutExpansionConfig(
        max_extra_steps=3,
        action_selector_kind=RolloutActionSelectorKind.FIRST_OPENABLE,
    ),
)
```

The default config is `one_ply`. Rollout action selector kinds are
`first_openable`, `random_openable`, and `no_rollout`. `random_openable` uses a
local `random.Random` instance seeded from rollout config and samples only among
currently openable branches. Built-in selectors keep expansion-only behavior,
but custom rollout action selectors can inspect legal, opened, and openable
actions to perform guided traversal before opening a frontier edge.

There are two rollout action-selector APIs. Serializable config is the right
surface for YAML and reproducible experiments:

```python
OpeningExpansionConfig(
    kind=OpeningExpansionKind.ROLLOUT,
    rollout=RolloutExpansionConfig(
        action_selector_kind=RolloutActionSelectorKind.RANDOM_OPENABLE,
    ),
)
```

Advanced callers can inject a `RolloutActionSelector` object at runtime through
`AlgorithmNodeTreeManager` or `create_opening_expansion_executor(...)`. The
precedence is:

1. explicit `opening_expansion_executor`
2. explicit `rollout_action_selector`
3. selector created from `RolloutExpansionConfig`

The object-injection API is intentionally separate from `RolloutExpansionConfig`
so serializable config stays YAML-friendly.

## Expansion Budgets

Search has two separate branch limiting layers. The existing initial-opening
limiter trims selector-proposed `OpeningInstructions` before expansion, so a
selector does not start more initial openings than the current branch limit can
plausibly allow. That pre-trim remains useful for planning initial openings.

Runtime opening expansion budgets are stricter: they are consumed by every
actual materialized tree edge immediately before it is opened. This includes
initial selector edges, rollout continuation edges, and existing-node
connections. Traversing an already-opened rollout edge does not consume branch
budget because it does not materialize a new edge. The budget is enforced inside
expansion executors because rollout length is only known while materializing the
path.

For example, with two branch slots remaining, a selector may ask to open
branches `[a, b]`. A rollout executor can spend one slot on `a`, spend the
second slot on a rollout continuation from `a`'s child, and then stop before
opening `b`. This keeps `tree_branch_limit` as a hard materialized-edge limit
without requiring the selector or pre-trim layer to predict rollout length.
