# Native prepared evaluation (initial supported subset)

This additive API prepares candidate-derived state independently of Growth and
installs it into an exclusively owned, cold-restored activation runtime. It has
no publication, model selection, scheduling, locking or activation authority.
It is an author implementation awaiting independent integration/review; it is not
a release or a production performance qualification.

## Operations and ownership

Import the public operations and immutable types from `anemone.prepared_evaluator`:

```python
source = freeze_evaluator_source(
    checkpoint, codec_type=InlineMorpionCodec, identity=source_identity,
    recipe=recipe, limits=limits,
)
prepared = prepare_evaluator_context(
    source, evaluator=candidate, evaluation_identity=candidate_identity,
    limits=limits, validate_direct_certificate=readonly_certificate_validator,
)
# Hold exclusive quiescent ownership throughout staging and installation.
# activation_search was cold-restored through the existing native loader from
# the exact same committed checkpoint, under the incumbent evaluator.
plan = stage_evaluator_context(
    activation_search, current_source=source, prepared=prepared,
    evaluator=candidate, evaluation_identity=candidate_identity, limits=limits,
)
report = plan.install()
```

The typed `SearchRuntimeCheckpointPayload` must come from a pinned, authenticated
native checkpoint. The caller verifies the committed artifact inventory and owns
its publication binding. Freeze copies it through the existing native writer's
serialization boundary, validates complete coverage and codec closure, reconstructs
cold native bindings and retains only immutable bytes, identities and recipe.
It also hashes those bytes and the complete cold-normalized runtime baseline.
An unsaved, in-memory payload is not a substitute for the committed native read
boundary: different stored state-summary presentation can correctly refuse staging.
No sorting or tolerant comparison is added by the prepared path.

`SourceIdentity` binds publication/inventory, incumbent, codec and Anemone build.
These are caller-authenticated opaque identities, not paths conferring trust.
`incumbent` accepts an opaque bootstrap string or a full `EvaluationIdentity` for
a previously installed candidate. The incumbent evaluator's `evaluation_identity`
attribute must match it. `EvaluationIdentity` binds bundle, weights, architecture,
encoder, exact state/input recipe, objective/score transform and the admitted
`cpu-fp32-strict-scalar-v1` numerical profile. Its identifiers must cover all
backend, dtype, thread and input settings relevant to predictions. The candidate
must expose that exact immutable identity. Anemone compares these identities; it
does not hash model tensors, configure Torch, or authenticate files on the caller's
behalf. Freeze model identity before preparation; do not mutate model weights under
a constant identity.

Preparation constructs fresh native nodes, state handles, resolver/cache, derived
objects and a private selector RNG. `ReadNodeStructure`/`DetachedStructure` preserve
native ordered navigation and reject structural writes. There is no
`TreeExploration`, tree manager, opening executor or runnable rollout in this arena.
Caller-provided codecs, evaluators and certificate validators are trusted Python
implementations; this is an ownership boundary, not a sandbox for hostile callbacks.
They must not close over live owner objects or share mutable decoded state.

The live full-refresh path and preparation invoke the same internal
`refresh_bound_context`: direct evaluation, reverse-depth/insertion backup,
certificate validation, full index computation and selector refresh. Existing
partial refresh paths remain unchanged. Certified terminal/direct values preserve
canonical provenance; unknown direct FORCED evidence refuses. A direct FORCED
validator must validate complete domain evidence, including its exact outcome,
without mutating state. Preparation consumes no search or rollout RNG draws and
rejects changes to already-loaded process Python/NumPy/Torch CPU RNGs. A failed
stochastic callback may have changed its own/global state; discard its helper,
not the untouched Growth runtime. Model construction precedes this API and retains
the caller's RNG-preservation responsibility.

Staging checks the actual owner recipe, complete ordered structure, codec handles,
incumbent values/proofs/caches, versions, expansions, selector and actual independent
RNG objects/states. It preallocates native evaluation/index objects bound to owner
structure and reconstructs the prepared selector with owner RNG/instructor bindings.
It performs no candidate inference. A plan retains only its staged objects and a
weak owner reference; it does not retain a second runnable search.

`install()` consumes the plan, rechecks identities and every owner precondition,
then assigns the staged objects and advances the evaluator version exactly once.
It preserves owner nodes, state handles/resolvers, graph links, opening executor
and both RNG objects. Failed preflight disposes staging and leaves the owner
unchanged. Any exception after mutation begins marks the runtime invalid and raises
`RuntimeInvalidated`: discard/reload it. Growth, evaluator mutation/refresh and both
native live checkpoint builders then refuse. Direct private-field access cannot be
made safe; there is no general rollback. `dispose()` is idempotent and releases an
unused plan without altering owner or durable state. Immutable source/context
storage is reclaimed by normal reference lifetime; there are no external leases.

Save/export only through the existing native checkpoint APIs after a successful
installation. There is deliberately no `PreparedEvaluatorContext.to_checkpoint_payload()`
and no prepared publication path. `InstallReport` records source fingerprint,
candidate identity, old/new version, installed coverage and RNG preservation; it
is not a durable acceptance receipt or a performance report.

## Supported configuration and bounds

V1 admits real Atomheart Morpion state adapters, a layered single-agent max DAG,
built-in `ExplicitMaxBackupPolicy`/`BestChildAggregationPolicy`/`MaxProofPolicy`,
either null indexes (`index_computation=None`) or `MIN_GLOBAL_CHANGE` indexes,
native node factories without cached
evaluator-side state representations, the native bound `SearchFactory.node_index_create`
with the same index recipe and no depth extension, branch-count stopping, all-children opening
and `ComposedNodeSelector(NoPriorityCheck, Linoo)` with initialized checkpointed
selector history. The recipe binds index computation, Linoo policy, branch limit and the full opening
configuration. One-ply uses the search RNG; supported random-openable or
random-legal-prefer-openable rollout configurations use a distinct independently
checkpointed RNG. Preparation never executes a rollout. Aliased RNGs, custom
executors/selectors/proof/index policies and missing histories refuse.

| `SupportedSearchRecipe.index_computation` | Required native state | Prepared support |
| --- | --- | --- |
| `None` | Every node index is exactly `None`; canonical null manager/factory | Supported |
| `IndexComputationType.MIN_GLOBAL_CHANGE` | Every node has `MinMaxPathValue`; canonical global-min-change manager/factory | Supported |
| Other indexes, string aliases, depth extensions or mixed coverage | Unsupported | Refused |

The prepared recipe defaults to `MIN_GLOBAL_CHANGE` for compatibility with the
original prepared API; callers using canonical null indexing must pass `None`
explicitly. This does not change any ordinary search default. The null path uses
the existing null factory/manager and optional-index checkpoint serialization.
It creates no placeholder `MinMaxPathValue` objects and treats legitimate absence
as complete evidence. Source, candidate context and actual owner must agree on
the recipe, including the bound factory used by subsequent Growth. Index absence
does not relax graph, proof, codec, Linoo history, version or RNG validation.

An admitted zero-argument inline codec class declares
`prepared_domain = "morpion-inline-v1"` and promises exclusively owned decoded state.
It must implement the existing incremental native codec. Saved data is admitted
before native atom restoration: only the built-in `valanga.game.SoloRole.SOLO`
enum atom is supported; other module/qualname-bearing enum atoms refuse. JSON
nesting is bounded at 256 levels. Ordinary native checkpoint APIs retain their
existing broader trusted-input contract. All anchor/delta parents
must be covered, acyclic and lower-depth. Codec ancestry may differ from graph
ancestry; native restoration remains authoritative. External state stores, reader
leases, arbitrary plugin factories and incomplete checkpoints are unsupported.
The initial public input is a materialized typed native payload; it is not a new
sharded ingestion API. Native sharded writers and loaders remain usable for installed
runtime export/continuation. No Chipiron codec or dependency was modified here.

`PreparationLimits` uses one absolute monotonic deadline, optional cancellation
predicate and positive node, branch, serialized-byte and state-decode ceilings.
Defaults are 512 nodes, 4096 branches, 16 MiB per source/derived artifact and 1024
state decodes per reconstruction. The same limits object may span all phases;
no phase renews its deadline. Source and derived byte limits are rechecked before
preparation/staging decoding even when artifacts were built under larger limits.
Decode allowances apply to each reconstruction, not
a cumulative multi-attempt reservation; the caller accounts attempts globally.
Checks occur around inference batches, during reconstruction/backup and between
canonical phases. Index/selector calls and arbitrary native inference cannot be
hard-interrupted by this cooperative API. Encoding/staging also need transient
allocations beyond serialized bytes. The caller must enforce hard process time,
RSS/output budgets, cleanup, and cumulative reservations. No GPU, stochastic model,
training or general numerical backend qualification is provided.

## Catch-up and cost

A context for an earlier head cannot be installed after Growth advances. Freeze the
complete current incumbent checkpoint and prepare a fresh context from that baseline.
A caller-owned evaluator cache may reuse **raw predictions only**, keyed by the full
candidate identity and exact canonical input. Node ID, state tag alone or matching
node counts are insufficient. Include depth in the input key if the encoder uses
it; canonical Anemone depth transformation still runs at the current depth.

Do not reuse old-head backups, proof results, PV/frontiers, index bounds, selector
heaps/history or evaluator versions. All derived computation and validation run
again against the current incumbent baseline, including newly reached nodes.
There is no generic delta journal or incremental context advance.

Freeze, preparation, staging and installation preflight all traverse the complete
source and allocate/serialize native records. Staging currently reconstructs another
detached arena and decodes the full closure to validate prepared records. The final
assignment loop is O(N), but `install()` also includes a complete graph/derived-state
fingerprint check. Cold activation restore, full catch-up math, staging and final
native save/export remain on the acceptance path. No constant-time handoff or net
speedup is established; independent qualification must measure stale-head hit rate,
all these costs and peak memory with the real model and intended checkpoint size.

## Focused evidence

`tests/test_prepared_evaluator*.py` generate fresh legal Morpion DAGs through native
Growth, compare detached/live full refresh, install into independently cold-restored
owners and compare exact native checkpoint bytes and subsequent nonzero Growth.
Coverage includes transpositions, alternate graph/codec ancestry, terminal/FORCED/
ESTIMATE, both null and global-min-change indexes, PV/selector payloads, both independent RNG streams,
raw-prediction catch-up, repeated candidates, both native shard layouts, bounded
refusals, stale plans, inference failure and failure after the first assignment.
The original refresh/partial-refresh and relevant checkpoint suites provide regression
coverage. These are focused author checks with deterministic scalar CPU fixtures,
not production lifecycle/model or performance acceptance. The historical J4.4d
140-comparison proof and its consumed invocation are not rerun.
