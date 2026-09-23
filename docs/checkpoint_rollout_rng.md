# Independent rollout RNG checkpoint continuation

Anemone 0.2.25 adds `SearchRuntimeCheckpointPayload.rollout_rng_state`, an
optional field containing `random.Random.getstate()` from the **live opening
executor's rollout action selector**. Both `random_openable` and
`random_legal_prefer_openable` own such an RNG. Runtime selector overrides
exposing a `Random` in `random_generator` use the same mechanism.

The existing `rng_state` still represents the search/node-selector stream.
The streams remain separate objects. Saving reads state without draws; restoring
calls `setstate` on the newly assembled executor's actual selector. Factories,
seeds, rollout actions, Linoo policies and uninterrupted execution are unchanged.
One-ply, first-openable, first-legal-prefer-openable and no-rollout configurations
have no rollout RNG state (`None`). A checkpoint containing rollout RNG state
cannot restore into a selector that has no RNG and raises a clear error.

The additive field is retained by dataclass JSON encoding (plain, gzip and zstd),
sharded metadata, materialized shard decoding, and streaming restore. Both RNG
state vectors are converted back from JSON lists to tuples before `setstate`.
Both `node_records` and `split` sharded layouts preserve it.

The checkpoint format stays at version 2 and the sharded format version is
unchanged. No required field or existing field meaning changed: the new field
has a `None` default, and shard readers use optional metadata lookup. JSON
adapters that construct this dataclass from its fields inherit the default.
Use 0.2.25 or newer readers for the new exact rollout-continuation guarantee.

Legacy checkpoints with the field absent or null remain readable. Their rollout
RNG retains the historical factory initialization from the configured seed.
Exact continuation of already-consumed legacy random rollouts cannot be
reconstructed, because those checkpoints never stored that stream's state.

Regression tests compare both random selectors before consumption, after one
choice, and after several choices across all formats, including unlimited
rollouts. They compare four subsequent operations: selected node/depth,
alternating Linoo subpolicy, actions actually chosen by the executor's selector,
node/branch counts, full tree payloads, rollout reports and both RNG states.
Legacy absence, deterministic configurations and saves during uninterrupted
execution are checked separately. Chipiron's real JSON adapter and production
sharded/streaming continuation are additionally verified in release integration.
