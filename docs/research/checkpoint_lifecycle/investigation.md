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
