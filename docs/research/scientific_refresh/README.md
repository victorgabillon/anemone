# Scientific evaluator refresh correction

Implemented on the exact approved Anemone base
`a07fb2fdd75e2020d9795d34ec1cf896eb86f3a0` (distribution 0.2.26).
GitHub protected `main` still pointed there when checked. The existing local
checkout was at `7950881` with six dirty tracked files; it was left untouched.
Implementation branch: `fix/scientific-refresh-correctness`, in the new ext4
worktree `/home/pompote/oldata/victor/morpion-scratch/anemone-scientific-refresh-20261024/worktree`.
Neither PR #77 nor #78 is included. Nothing is merged or published.

Research input: Chipiron commit `60fdf85bb26e34f7d502122f34f3208ae2fc19d9`,
`docs/research/morpion_lean_scientific_model_activation.md` and its executable
`model_activation/` oracle/witnesses. The new regression fixtures use Anemone and
Atomheart directly, with no Chipiron import or production change.

## Defects and corrections

1. `NodeDirectEvaluator.evaluate_nodes()` cleared every nonterminal direct Value
   before batching predictions. Model refresh could replace a valid FORCED(9)
   with ESTIMATE(42), also overwriting its evaluator-version provenance. Refresh
   now explicitly preserves exact **direct** Values and their original objects
   and versions. ESTIMATE/missing nonterminal direct Values are evaluated even
   when the node's tree/effective Value is exact. Certainty predicates and proof
   classification reuse the canonical implementation.
2. `ValuePropagator` decided whether to schedule a changed node's parents from
   its local tree-backup/PV result. Its direct Value was already changed before
   that comparison, so an effective-value change could disappear. The new path
   settles changed nodes deepest first and conservatively seeds **every incoming
   parent** from known local changes. Subsequent ancestors still use existing
   incremental depth waves; ordinary updates do not trigger a full graph backup.
   Local direct changes retain certified tree Values; ancestor recomputation
   refuses a certificate it cannot reconstruct rather than silently downgrading it.

The existing `refresh_with_evaluator(scope="all")` is consolidated into bounded
windows (default 64), a complete deepest-first backup of every opened node, and
one exploration-index/selector refresh. No alternate activation kernel, persistent
cache, duplicate graph, proof registry, objective, or numerical policy is added.

## Explicit certificate and refresh contract

```python
runtime.refresh_with_evaluator(
    new_evaluator,
    scope="all",
    batch_size=64,
    validate_direct_certificate=validate_original_proof,
)
```

The optional callback is `Callable[[NodeT], bool]`. For each direct FORCED Value,
it must establish genuine proof validity for the same state and objective and
bind the stored score to its original proof authority. It must be read-only and
may be called both before and after inference. A missing/false validator refuses
before changing the evaluator. A FORCED enum or evaluator-version number alone
is not proof evidence. Tests use an independent solver over the complete tiny
game, including unopened actions, plus the original certificate identity.

TERMINAL direct Values require the node's own terminal state and canonical
metadata, and retain their original Value/version. Structural terminal skipping
retains the existing reevaluation policy. ESTIMATE outputs remain estimates;
effective exactness is never copied to a direct prediction.

Tree certificates must be reconstructible under the existing built-in explicit
best-child max/minimax backup and proof rules. The preflight uses canonical
comparison/classification without invoking backup or modifying the tree Value.
Unknown policy implementations/subclasses and unverifiable partial tree proofs
are refused. Reconstructed existing tree certificates must remain equal during
complete refresh, and their original Value objects are retained. Contradictory
same-state certificates, or effective selection that hides a direct proof, refuse.
A valid partial minimax proof is still governed by its existing minimax proof rule;
no max rule is substituted for it.

The new `skipped_certified_count` reports preserved nonterminal direct certificates.
`reevaluate_nodes()` also preserves certified direct Values. Deliberate replacement
remains available through `NodeDirectEvaluator.evaluate_nodes()` with its default
`preserve_certified_values=False` and the explicit scalar update API. Removing a
child's exactness cannot silently revoke an ancestor's tree certificate: the
propagation now raises when that proof cannot be reconstructed.

Calls require a quiescent runtime. Known invalid proofs/policies fail in preflight
before evaluator replacement. Inference errors or newly contradictory predictions
can fail after mutation; the runtime must then be discarded/reloaded. This is a
scientific refresh operation, with no transaction, locking, rollback or publication
protocol. Partial scopes remain partial; only `scope="all"` promises a complete pass.

## Before and after evidence

| Witness | Approved unmodified baseline | Fixed regression |
| --- | --- | --- |
| Independently proven unique terminal score 9, neural prediction 42 | Direct becomes ESTIMATE(42) | Same direct FORCED(9) object/version retained |
| Actual seven-node Morpion, only node 2 changes to raw 100 | Root stays `2.2600000452000004` | Root is exactly `100.00000100000001` |
| Same seven-node graph, ordinary direct-value update | Same stale root | Same correct root; node 2 tree Value/PV unchanged |
| Actual one-action move-34 Morpion endgame | Research distinction preserved as required | Direct ESTIMATE(42), tree/effective FORCED(35), terminal TERMINAL(35) |
| Unverifiable partial tree FORCED(35) | Stock refresh overwrites proof | Explicit refusal before changing evaluator or Values |

The real graph is regenerated with seed 4, branch limit 6, two growth steps, the
same symmetry-reduced legal actions, and the original `moves + .01 * legal_count`
heuristic. The test asserts the exact six edges as well as the score. The existing
factor `(1 / 0.99999999) ** tree_depth` is retained bit-for-bit. The forced-nine
fixture is a different, fully specified tiny game; it makes no false claim about
the real initial Morpion board's optimum.

`tests/scientific_refresh_helpers.py` implements an independent small-fixture
max/effective/PV/order oracle. It calls no Anemone backup, propagation, effective
selection, proof or ordering helper. Comparisons include complete direct/tree/
effective Values (therefore their independent certainties and metadata), selected
source, principal variation, all child ordering, terminality and exactness.

The 32 scientific regressions cover all twelve requested cases, including multiple
parents, unchanged local tree/PV, exact/estimate ties, effective-source/certainty
changes at an unchanged scalar score, full opening, missing direct Values,
42→7→42 refreshes with windows 1/7/64, and idempotent predictions. A real generated
Morpion DAG has at least seven shared nodes. Refresh preserves all edge directions,
state identities, opening state, RNG, selector parity and latest search reports.
A generic refreshed checkpoint reload preserves Values, provenance versions and the next
three selections/RNG states when the committed evaluator is supplied at load time.

## Validation

- Baseline red tests: **3 failures**, each asserting required correct behavior.
  Source was extracted from the approved commit into a separate scratch directory;
  production edits began only after the initial red reproduction. A later repeat
  with the independently proven tiny-game certificate also fails all three.
  See [baseline source](baseline_red_tests.py) and [recorded failures](baseline-red.log).
- Fixed independent scientific regressions: **32 passed**.
- Full suite: **932 passed, zero failures/errors/skips**, 16.90 seconds, including
  all 900 existing tests: Value/certainty/backup, checkpoint, selector, debug and
  other integrations. No historical checkpoint data or scientific expectations
  were weakened. Test doubles were extended for the new optional evaluator
  argument, report count, and canonical tree-Value surface.
- Ruff source/new Python checks and format checks pass; mypy passes all 288 source
  files; pyright reports zero errors/warnings; pylint passes at 10.00/10.
- Both wheel and sdist build successfully; Twine accepts both distributions.

[Machine-readable validation](validation.json) records every test-module count
and raw log/JUnit locations and hashes. Raw logs and the isolated Python 3.13.5
environment are retained alongside the worktree. Main commands, from this worktree:

```sh
../venv/bin/python -m pytest --junitxml=../full-suite.xml
../venv/bin/python -m mypy --config-file pyproject.toml src/anemone
../venv/bin/python -m pyright --project . src/anemone
../venv/bin/python -m pylint src/anemone
../venv/bin/python -m ruff check src
../venv/bin/python -m ruff format --check src tests
../venv/bin/python -m build
../venv/bin/python -m twine check dist/*
```

To repeat the baseline-only red witnesses, extract approved `src/` into
`../baseline/` and run with `PYTHONPATH=../baseline/src:.` and the same interpreter,
using `-m pytest docs/research/scientific_refresh/baseline_red_tests.py --no-cov`.
The fixed tests supply explicit proof validation through the corrected API.

## Production size and bounded performance

Six production files: **+237 / −44 lines, net +193**, including comments/blank
lines. Line edit classification: 154 pure inserted lines; 44 existing lines
replaced by 83 lines; zero separate pure deletions. The propagation change removes
its old filtering helpers. [Per-file sizes and definitions](code-size.json).

Fresh synthetic ternary trees, six passes each (one warmup, five measured), serial
fixed and baseline processes, default window 64; no 10k benchmark. Refresh timing
uses `tracemalloc`, ordinary scalar-update timing does not. Peak allocation is
Python allocation during refresh, including retained changed Value/cache objects,
not total RSS or a guarantee about decoded states. These are small synthetic
component measurements, without neural inference.

| Nodes | Baseline refresh, ms | Fixed refresh, ms | Baseline / fixed peak, KiB | Baseline / fixed ordinary update, ms |
| --- | ---: | ---: | ---: | ---: |
| 40 | 13.76 | 6.52 | 16.7 / 64.1 | 0.418 / 0.408 |
| 364 | 126.04 | 60.08 | 151.2 / 93.2 | 0.697 / 0.697 |
| 1,093 | 386.07 | 182.98 | 368.7 / 270.1 | 0.916 / 0.906 |

The fixed small-tree peak includes its window bookkeeping. Ordinary update costs
show no material difference at this scale; they also include the existing public
ID lookup and index refresh. Baseline timing is not evidence of correctness: its
scientific witnesses remain red. [Driver](measure.py),
[baseline measurements](baseline-performance.json), [fixed measurements](fixed-performance.json).
Run the driver as `python -m docs.research.scientific_refresh.measure`, selecting
baseline source with the same `PYTHONPATH` mechanism if needed.

Extra refresh working storage is bounded by the window, depth registry traversal
and one parent's outgoing degree, in addition to existing graph/model/caches and
stored PVs. Backup/order work still follows existing edge traversal and sorting.
The checkpoint resolver may still retain all decoded states.

## Remaining blockers and integration strategy

This patch does **not** make production model activation ready. Exact status:

- Decoded-state retention: resolver unchanged; bounded query windows do not bound
  its `_resolved_states` cache. PR #77/#78 and their worktrees remain separate.
- Restore evaluator versions: `set_evaluator()` still increments on every call.
  Checkpoint loading already accepts the evaluator; callers should provide the
  committed model there. No resume API or version semantics is changed here.
- Representative-parent checkpoint round-trip discrepancy and restored child-link
  insertion-order differences: unchanged; no payload/codec/data normalization.
- Growth versus Reevaluation scaling: unchanged; this patch preserves Growth's
  exact depth factor. A future integration must choose and enforce compatibility.
- Pending producer publication race: untouched in Chipiron; the final producer
  identity check/publication still needs its shared ownership boundary and race test.
- Atomic accepted-model binding, recovery and the M2 crash/concurrency matrix:
  unimplemented by this Anemone correction.

Recommend reviewing this isolated commit directly against the approved base,
then releasing a new Anemone patch version (recommended **0.2.27**, subject to the
maintainer's release sequence) rather than replacing published 0.2.26 artifacts.
Chipiron should pin that release or exact reviewed commit, rerun its scientific
integration suite, and explicitly request `scope="all"`, bounded windows and a
real certificate validator. Resolve the separate retention, resume, scaling,
publication and atomic activation gates before enabling any production transition.
No real gen38 tree, historical experiment, production bootstrap, worker, model
activation, checkpoint migration, merge or Chipiron production edit was performed.
