# Debug toy scenarios

The toy scenarios are tiny deterministic domains for the Anemone debug GUI.
They exist so you can inspect a known tree step by step while still exercising
the real Anemone exploration, selection, backup, and debug pipeline.

They are useful for validating:

- tree growth and child linking
- backed-up value changes
- principal variation updates
- minimax behavior
- deceptive shallow evaluation that flips after deeper backup

These scenarios are safe playgrounds because the domain is tiny and fully
declarative, not because the engine is fake. Each run goes through the normal
live debug environment and produces a standard session directory for the
browser UI.

The scenarios intentionally use one simple, explicit production search setup:

- uniform node selection
- all-children opening
- a tree-branch-limit stopping criterion sized to the declared toy graph
- a softmax recommender after exploration

That keeps the runs deterministic and easy to inspect while still exercising
the real engine path.

## Included scenarios

### `single_agent_backup`

Tree:

- `root`
  - `A`
    - `A1 = 3`
    - `A2 = 8`
  - `B`
    - `B1 = 5`
    - `B2 = 6`

Expected outcome:

- root backed-up value: `8`
- root PV: `A -> A2`

This is the smallest scenario for inspecting straightforward max backup from the
real engine. It uses the same minmax-capable tree-evaluation family as the
production Tree-and-Value path, but configured with a single-agent max objective
so the numeric `8 > 6` comparison remains faithful.

### `minimax_micro`

Tree:

- `root` is MAX
  - `A` is MIN
    - `A1 = 4`
    - `A2 = 7`
  - `B` is MIN
    - `B1 = 2`
    - `B2 = 9`

Expected outcome:

- `A` subtree value: `4`
- `B` subtree value: `2`
- root value: `4`
- root PV: `A -> A1`

This is the smallest adversarial backup scenario running through the real
minimax stack.

### `deceptive_trap`

Tree:

- `root`
  - `A` has shallow heuristic `9`, but deeper terminals are `1` and `0`
  - `B` has shallow heuristic `5`, but deeper terminals are `6` and `4`

Expected outcome:

- root final value: `6`
- root final PV: `B -> B1`

This scenario is useful for inspecting value flips, `BackupFinished` changes,
and "next backed-up value change" navigation in the GUI while still using the
production backup path.

### `minimax_semantic_stress`

Tree:

- `root` is MAX
  - `A` is a solved MIN branch with exact terminal children
  - `B` is a partial MIN branch with one exact child and one heuristic child
  - `C` is another partial MIN branch in a loss-leaning shape
  - `D` is a solved losing MIN branch

Expected outcome:

- useful GUI contrast between `TERMINAL`, `FORCED`, and `ESTIMATE`
- no single deterministic final root assertion is required

This scenario is a semantics demo first. It is useful when you want one tree
that visibly mixes solved interior nodes, unresolved interior nodes, and exact
terminal leaves.

## Running the examples

Run any scenario directly:

```bash
python examples/debug_scenarios/run_single_agent_backup.py
python examples/debug_scenarios/run_minimax_micro.py
python examples/debug_scenarios/run_deceptive_trap.py
```

Or use the convenience runner:

```bash
python examples/debug_scenarios/run_all_scenarios.py
python examples/debug_scenarios/run_all_scenarios.py deceptive_trap
```

Each script prints the session directory and the recommended serving command.
The examples use a toy domain, but they do not use a toy exploration engine.

## Serving the GUI

The examples finalize the session after the run and then print a serving
command. Run that command from the repository root, typically in another
terminal:

```bash
python -c "from anemone.debug import serve_live_debug_session; serve_live_debug_session('runs/debug_scenarios/deceptive_trap', port=8000)"
```

Then open the printed local URL in the browser.

## Launching toy scenarios from the browser root

If you want the browser itself to launch built-in toy scenarios, start the
browser root instead of a single precomputed session:

```bash
PYTHONPATH=src:. python -c "from anemone.debug import serve_debug_browser; serve_debug_browser('runs/debug_browser', port=8000)"
```

Open the printed local URL, choose a built-in scenario from the dropdown, and
click `Run scenario`. The browser will switch to the generated live session
under `runs/debug_browser/sessions/<scenario-name>/`.

## What to inspect in the GUI

For all scenarios:

- timeline entries for iteration boundaries and backup events
- graph growth in the snapshot view
- node inspector values for direct and backed-up values
- principal variation in the root node
- the real `NodeSelected`, `ChildLinked`, `DirectValueAssigned`, and
  `BackupFinished` event flow from the production stack

Recommended focus by scenario:

- `single_agent_backup`: verify root value and PV settle on `A -> A2`
- `minimax_micro`: verify MIN children collapse their subtrees before the root
  chooses between them
- `deceptive_trap`: step through backup changes until the preferred branch flips
  from `A` to `B`
