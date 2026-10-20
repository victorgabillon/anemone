# Committed live-checkpoint lifecycle investigation

Protected base: Anemone `a07fb2fdd75e2020d9795d34ec1cf896eb86f3a0`. Baseline integration uses clean PR77 `ec9fec7e47ca661eb1dd9330d115d010d9e5e64d` plus unchanged Chipiron PR74 `aeece2c1694a90bb79880db1733966d22c36c6a1`. Production warm reuse remains disabled; those worktrees/PRs are not modified.

## Exact baseline failure

The deterministic 10k fixture again starts at 10,008 nodes and ends at 10,110 nodes / 11,904 branches. The retained generation-3 and generation-4 checkpoints differ at exactly **one** node, ID **7952**, depth **4**. Only two fields differ:

| Field | Cold | Experimental warm |
| --- | --- | --- |
| `parent_node_id` | 412 | 417 |
| `branch_from_parent` | `(1, -2, 1, 4)` | `(1, 4, -2, 4)` |

Depth, state summary/tag `4402156646146418305`, opening status and all other shell fields agree. Metadata/RNG, state payloads, node runtime/values/exactness, selector, latest expansions and selected nodes agree exactly. Both states are checkpoint-backed **anchors**, with no persisted delta parent or delta branch. Full incoming edge sets agree: parent 412 via `(1,-2,1,4)` and parent 417 via `(1,4,-2,4)`.

Generation 1 has one parent (417) and a materialized state at initial save. Both first restore it to a checkpoint-backed anchor. The first Growth step selects parent 412 and adds its edge to existing node 7952. Generation 2 therefore has incoming order `[417,412]` in both runs and saves representative 417. The next cold restore replays persisted child records in checkpoint node order, giving incoming order `[412,417]`; warm retains `[417,412]`. `_representative_parent_link` falls back to first runtime parent for an anchor. The generation-3 mismatch follows directly and persists at generation 4. This is not a state-handle type difference in the original failure.

`baseline_node_shell_differences.json` records every differing field and complete parent/branch/state-handle context. Original retained PR74 benchmark evidence is unchanged. New raw fixtures are under `/tmp/anemone-lifecycle-baseline-20261020`.

## Minimal independent causes

A six-node initial checkpoint in the fake YAML domain (eight-node domain, no Morpion code) includes a DAG join and two branches from the same parent. Tests isolate:

- A live graph saved twice is stable.
- An anchor with incoming storage order `[3,2]` saves representative 3. Cold restore replays child records as `[2,3]`; resave chooses 2. Changing only storage order reproduces it. Keeping the same order while changing only handle type does not change this anchor case.
- A valid first-parent delta keeps the same representative across restore despite changed incoming storage order, because the restored delta carries its chosen parent.
- If the codec rejects the first parent and delta encoding chooses the other valid parent, the materialized save's representative and state ancestry can differ. Merely replacing the handle with its checkpoint-backed delta changes the representative, even with identical parent order. This is a second, independent cause relevant to newly saved handles.

`baseline_minimal_cases.json` records A/B/C/D/E/F full-content comparisons. These are ordering/reference lifecycle differences with equal graph edge sets; equal graph topology alone does not satisfy checkpoint parity.

## Proposed scope

An explicit Anemone-owned adoption boundary should reproduce existing cold restoration on the same nodes: replay its incoming-parent ordering and adopt the saved lazy handles using existing payload stores/resolver helpers. Preserve all graph edges, node objects, values, selector, RNG and external handle/cache references. Do not change checkpoint format, first-save serialization policy, or existing readers. A compact prepare/adopt plan is needed so the caller can finish durable checkpoint/run-state publication before mutation without retaining the complete typed node-runtime payload graph.

Legacy anchor save/restore/save can already change the representative once; changing that general serializer/reader policy is outside this patch. The intended parity is the existing cold path, including that behavior. Subsequent adopted checkpoint round trips must be idempotent. Any broader policy change requires separate review.

## Lifecycle API

`prepare_checkpoint_adoption(search, payload, state_codec=codec)` validates this runtime's just-built payload and prepares only a weak runtime reference, scalar signatures, IDs and a shared lazy payload resolver. It retains no AlgorithmNode/TreeExploration objects or evaluation/runtime payload graph. The caller can drop the full typed payload after writing it, finish its own durable generation publication, and then call `plan.adopt()` once. On save/publication failure, drop the plan. Nothing happens automatically.

Adoption validates that the same graph and RNG/evaluator version remain current, allocates lazy handles, reorders multi-parent dictionaries in place according to checkpoint node order (the native loader's linking order), and assigns the saved handles. The dictionary and branch-set objects remain attached to existing views. Old handles and resolver caches are never cleared; external owners remain valid. All graph nodes/edges, values, exactness, evaluator, selector object/state and RNG are retained. The existing caller's selector checkpoint-rebind policy remains separate; this API does not refresh values or reset selector caches.

No checkpoint schema, first-save representative selection, or historical reader is changed. Preparing/adopting is O(N+E) compact metadata work with zero state decodes. It is not a new loader for unrelated/untrusted opaque codec payloads: supply the unmodified just-built payload, retain codec refs immutably, and keep the runtime quiescent across the commit boundary.

```python
payload = build_search_checkpoint_payload(search, state_codec=codec)
plan = prepare_checkpoint_adoption(search, payload, state_codec=codec)
write_checkpoint_json_payload(payload, checkpoint_path)
del payload
publish_generation_and_run_state()  # caller-owned, durable publication
stats = plan.adopt()                # never call if publication failed
```

The API is independent of PR77 and based directly on protected main. Integration will use a separately pinned temporary commit combining this independent change with PR77; neither existing PR/worktree will be modified. Production warm Growth remains disabled.

## Isolated Chipiron integration and the eviction owner

The original 10k case is resolved in the isolated adapter: 10,008 initial nodes, 10,110 final nodes and 11,904 branches. All three saves have identical complete scientific component hashes and selected-node sequences; every decompressed record in retained generations 3 and 4 compares equal, including **zero differing node shells**. This holds with export skipped and with full training export enabled. No parent field is normalized or ignored. Cold restores three times; warm restores once and keeps the same runtime.

Active eviction initially revealed a **separate caller-owned lifecycle counterexample**. The native handles/order adoption fixed shells, but retaining Chipiron's `_LiveCompactStateResolver.state_payloads_by_node_id` across saves let warm encode deltas against old compact parents. Cold `load_or_create` resets that table, causing anchor fallbacks instead. Generation 3 had 15 differing state-payload records; generation 4 had 34. For example, node 1038 is a cold anchor and warm delta. All other components and selections agree. Full raw differences are retained in `eviction_owner_differences.json`; the original failed fixture remains untouched at `/tmp/anemone-lifecycle-eviction-1k`.

The experiment therefore calls Chipiron's existing `_reset_growth_state_eviction_runtime()` **after committed native adoption**. This rotates the caller's resolver owner, metrics, scan cursor and recent-selection history to its existing cold boundary. It never clears an old resolver or cache: any external handle can still resolve through its old owner. This is an integration requirement, not Anemone knowledge of Chipiron policy. Anemone adoption alone cannot reset caller-owned eviction tables. The implementation PR contains no Chipiron runtime change.

With that explicit rotation, the final 1k eviction, rollout, and combined cases pass all components and all three saves. These use `frontier_cold`, `delta_when_safe`, recent window 1, scan interval 1, maximum delta depth 32, bounded cache 128, and rollout `max_extra_steps=2`. A separate cache validation uses bound **8**, explicitly decodes compact states before saves, checks immediate memoization and the bound, then checks held external handles after adoption/rotation. Its complete checkpoints and selections match. Cold and warm have exactly the same 363/192/259 cache misses and 355/184/251 evictions across the three boundaries, with cache size always 8. This forced churn adds no warm-specific replay cost. These reads occur outside prepare/adopt; all lifecycle operations still decode zero states.

## Measurements

Each measured mode runs in a fresh CPU-only process, sequentially. Numbers include opt-in boundary instrumentation and are single samples, not throughput estimates for production. All entries use three saves and cold/warm restore counts 3/1. RSS is MiB; peak is process-lifetime `ru_maxrss`, not a stage-local peak.

| Fixture | Final nodes / branches | Cold / warm elapsed s | Cold / warm ending RSS MiB | Cold / warm process peak MiB | Checkpoint parity |
| --- | --- | --- | --- | --- | --- |
| 10k_without_export | 10110 / 11904 | 9.051 / 8.602 | 376.8 / 382.4 | 455.9 / 455.5 | exact |
| 10k_full_export | 10110 / 11904 | 157.727 / 189.127 | 547.8 / 491.8 | 552.0 / 550.5 | exact |
| 1k_eviction | 1108 / 1196 | 2.702 / 3.437 | 311.6 / 311.7 | 311.2 / 311.4 | exact |
| 1k_rollout | 1267 / 1322 | 3.735 / 4.583 | 321.5 / 321.5 | 323.0 / 323.3 | exact |
| 1k_eviction-rollout | 1267 / 1322 | 4.094 / 5.051 | 322.3 / 322.2 | 322.7 / 322.6 | exact |
| 1k_cache-bound | 1267 / 1322 | 5.430 / 6.633 | 322.0 / 323.0 | 322.5 / 322.6 | exact |

In the original 10k no-export run, prepare costs 0.129–0.130 seconds and adoption 0.090–0.094 seconds per commit. Each binds approximately 10k lazy handles and reorders 4, 6, then 1 multi-parent records. No state is decoded. Checkpoint wrapper save costs are 1.61–1.71 seconds cold and 1.88–2.05 seconds warm, including the extra preparation/probes. Per-stage observations and all individual measurements are in `integration_results.json`.

Full export has exact parity but warm is approximately **20% slower** in this sample. Cold export stages take 46.64, 48.33 and 53.42 seconds; warm takes 58.33, 58.74 and 61.84 seconds. Native prepare/adopt costs together remain below 0.37 seconds per boundary. The export time difference is observed; its cause is not established here. There is no demonstrated end-to-end speedup with export and no claim of production memory efficiency.

The two 10k timing runs preceded the later caller-owner rotation and optional internal/cache probes; eviction policy is `none` in both. Final 1k cases use the final adapter. These versions and exact source identities are recorded in the JSON rather than silently treating all instrumentation versions as identical.

A separate intrusive 10k object probe observes exactly one live `TreeExploration` and unchanged `AlgorithmNode`/`TreeNode` counts throughout each prepare/adopt (10,049; 10,078; 10,110). At handle-allocation snapshots, tracked objects temporarily increase by **N+32**, matching new lazy handles and small metadata containers. Prepare resolver snapshots add at most 587 tracked objects. No additional nodes/runtime are constructed. These are sampled GC object counts, not exact allocator peaks; intrusive snapshot timings are excluded from comparisons. Old resolver caches remain owned and valid, and adopted handles share a new initially empty native resolver. Native regression replay decodes each node at most once and reuses its result.

## Source identities and compatibility

- Protected Anemone main, checked before branching and again before publication: `a07fb2fdd75e2020d9795d34ec1cf896eb86f3a0`, protected `true`.
- Independent native implementation: `e29b58402dcc0c3473c910ee58de04cf212cb26a`, branch `perf/checkpoint-live-state-lifecycle`, worktree `/home/pompote/oldata/victor/anemone-checkpoint-lifecycle`. Subsequent PR commits retain research evidence only; the three changed native source files are byte-identical to the tested combined checkout.
- Unchanged PR77: `ec9fec7e47ca661eb1dd9330d115d010d9e5e64d`, import `/home/pompote/oldata/victor/anemone-state-free-compare/src/anemone/__init__.py` for the baseline.
- Clearly labeled temporary PR77 + native implementation combination: `866560f7606eea095981ec25090c5f2c5908a643`, import `/tmp/anemone-checkpoint-lifecycle-pr77/src/anemone/__init__.py` for corrected integration. This combination is not the native PR branch and is not a dependency bump.
- Unchanged Chipiron PR74: `aeece2c1694a90bb79880db1733966d22c36c6a1`. Used unchanged for the 10k baseline/no-export correction.
- Temporary Chipiron experiment adapter: `027231820f1b2a192d1adb4ec0d90469dac19dbe`, import `/tmp/chipiron-checkpoint-lifecycle-integration/src/chipiron`. Its tracked diff (`chipiron_experiment.patch`) only removes old experimental eviction/rollout cold gates; ordinary `_warm_reuse_is_validated()` still returns **False**. Only the benchmark subclass explicitly opts in. This commit is not pushed and does not update PR74.
- Python/package identities are in `environment.json`. The Anemone distribution remains version 0.2.26; explicit `PYTHONPATH` selects the pinned source. Core source SHA256 fingerprints, raw result fingerprints and the final executed adapter fingerprint are in `integration_results.json`. The retained driver is the Ruff-formatted equivalent of the exact executed text retained at `/tmp/benchmark_anemone_lifecycle.py`.

The complete 10k fixture is an existing old checkpoint copied from a newly generated synthetic baseline, not rewritten in place. It and the native legacy-format-2-without-summaries regression demonstrate compatibility within the tested scope. No real gen38 restore or historical checkpoint migration is claimed or attempted. Both codecs' anchor/delta ancestry and same-parent alternate branches are tested; equal graph topology is never accepted as a substitute for checkpoint equality.

Tree evaluation retains its `TreeNode` and accesses its current state handle dynamically; adoption does not leave an evaluation-owned old handle behind. Caller-held old handles/resolvers remain legitimate independent owners. No cache is forcibly cleared, and no second Chipiron checkpoint resolver is introduced.

## Validation and reproduction

All standard tox environments pass: **924 tests**, Ruff checks/format, Pylint 10.00/10, mypy (288 files), pyright (zero errors/warnings), sdist/wheel and Twine checks. The 24 new parameterized regressions cover both failing-before causes, repeated exact continuation, stabilized save/restore/save, multiple parents and branches, codec ancestry, zero-decode adoption, memoized replay, stale/interrupted lifecycles, weak runtime ownership, retained cache/edge owners, Linoo/RNG, two rollout RNG policies, reevaluation value updates and legacy fields. `validation.json` records the exact command and raw log fingerprint. GitHub CI is reported on the draft PR; local results are not represented as remote CI.

The generic tests run directly on the native branch without PR77:

```sh
python -m pytest tests/test_checkpoint_adoption.py
tox -p2 -o --parallel-no-spinner
PYTHONPATH=src:. python docs/research/checkpoint_lifecycle/minimal_baseline.py --output /tmp/minimal-lifecycle.json
```

For a new synthetic integration run, check out the pinned PR74 into a new `/tmp` worktree, apply `chipiron_experiment.patch` only there, and use a separate clean Anemone checkout combining pinned PR77 and implementation commit `e29b584`. The tested combination is identified above; locally reconstructed commits can have different commit metadata, so pass their actual full SHAs explicitly and verify the recorded native source hashes. Use the isolated environment in `environment.json`, never a historical workspace or a protected checkout. For example, with `CHIPIRON_LIFECYCLE_SOURCE`, `PYTHONPATH`, and the two SHA arguments pointing at those isolated sources:

```sh
PYTHONDONTWRITEBYTECODE=1 python docs/research/checkpoint_lifecycle/benchmark_integration.py \
  --root /tmp/new-lifecycle-fixture --nodes 1000 --mode prepare \
  --config eviction-rollout --anemone-sha ANEMONE_FULL_SHA --chipiron-sha CHIPIRON_FULL_SHA
# Repeat with --mode cold, then warm, then compare; use the same flags/root.
# --export enables full training export; --exercise-cache forces the bound-8 check.
# --object-probe enables intrusive object snapshots, excluded from timing claims.
# --retain-eviction-owner reproduces the separate eviction failure; compare must fail.
```

The driver rejects dirty/mismatched sources and non-temporary output paths, and measurement inputs must have the synthetic fixture marker. To reproduce the original parent-order failure, use unchanged PR74 and pinned PR77 with `diagnose_baseline.py` (`--mode prepare`, `cold`, `warm`, `compare`, same new `--root`, `--nodes 10000`). This diagnostic intentionally does not adopt or weaken comparison. Existing raw evidence remains under the `/tmp` roots recorded in JSON; no original retained benchmark is overwritten.

## Remaining production blockers

The concrete native lifecycle change and bounded parity results justify a **draft**, independently reviewable Anemone PR. They do not authorize production warm reuse. Chipiron still needs separate review/authorization for native adoption after durable publication, caller-owned eviction rotation, and its existing selector rebind. Full export is slower in the measured warm run. The tests do not establish large-tree, unlimited-rollout, long-running, model-switch, or four-worker asynchronous safety/performance. Existing legacy first-round-trip parent policy remains unchanged and requires separate review if it is to become globally canonical.

The original dirty Chipiron workspace, PR74 and PR77 worktrees, original retained benchmarks, historical/derived runs and dependency pins are unchanged. No real Growth experiment, gen38 restore, worker daemon, training run or merge was launched.
