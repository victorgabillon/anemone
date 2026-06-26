# Sharded runtime checkpoints

This note describes the planned generic Anemone sharded runtime checkpoint
format. It is a design and skeleton target, not a replacement for the current
monolithic checkpoint path.

## Problem

The current runtime checkpoint path writes one compressed JSON payload and
restores it in one shot. Restore currently materializes:

- one full raw decoded JSON dict/list graph,
- one full typed `SearchRuntimeCheckpointPayload` dataclass graph,
- then the final live runtime tree.

On large trees this creates high peak RSS. After temporary raw and typed
checkpoint graphs are dropped, CPython may retain arenas, so process RSS can
remain far above the final reachable runtime graph. C1 measurements on a 42k
Morpion checkpoint showed the first large RSS jump at `json.load`, a second
jump during typed payload construction, and high retained RSS after GC.

## Goals

- Keep the format generic to Anemone.
- Keep domain state payloads opaque and codec-owned.
- Reduce restore peak memory by loading bounded shard-sized graphs.
- Support future streaming or incremental restore.
- Preserve correctness and equivalence with the current checkpoint semantics.
- Keep selector state and latest expansion semantics exact.
- Keep a manifest that is stable, portable, and independent of absolute paths.

## Non-goals for C2a-C2c

- No default behavior change.
- No migration from existing monolithic checkpoints.
- No streaming sharded restore path yet.
- No streaming JSON parser dependency.
- No Morpion-specific optimization in Anemone.
- No claimed memory savings from the writer or non-streaming reader alone.

## C2b Writer Layout

C2b writes a first-stage node-record shard layout. Each node record contains
the same structural, state, linked-child, evaluation, and exploration-index
payload fields currently stored in one `AlgorithmNodeCheckpointPayload`.

```text
checkpoint_sharded/
  manifest.json
  metadata.json.zst
  node_records/
    nodes_000000.jsonl.zst
    nodes_000001.jsonl.zst
  selector/
    selector.json.zst
  latest_expansions/
    latest_expansions.json.zst
```

This layout is intentionally simpler than the long-term split layout. It is
generic, easy to test, and compatible with a future reader that processes a
bounded number of nodes per shard before dropping each raw shard.

## Split Layout

C2e adds an experimental split layout that separates state payloads from node
metadata and runtime records. This layout is intended for the incremental
runtime reader; the C2c typed-payload reader remains a node-record layout
equivalence tool.

```text
checkpoint_sharded/
  manifest.json
  metadata.json.zst
  node_shells/
    nodes_000000.jsonl.zst
    nodes_000001.jsonl.zst
  state_payloads/
    state_payloads_000000.jsonl.zst
  node_runtime/
    node_runtime_000000.jsonl.zst
  selector/
    selector.json.zst
  latest_expansions/
    latest_expansions.json.zst
```

The exact shard names can evolve, but manifest shard paths must remain relative
POSIX paths contained inside the checkpoint directory.

## Manifest

The manifest should include:

- monolithic checkpoint format version,
- sharded checkpoint format version,
- generation when the caller has one,
- intended node count per shard,
- total node count,
- total branch count when known,
- shard references with kind, relative path, record count, encoding, byte sizes,
- optional checksum fields such as `sha256`,
- encoder/compression metadata.

The C2a-C2c implementation provides:

- `ShardedCheckpointManifest`,
- `ShardedCheckpointShardRef`,
- manifest JSON conversion,
- manifest read/write helpers,
- relative shard path validation.
- a generic `write_sharded_search_checkpoint(...)` writer from a fully built
  `SearchRuntimeCheckpointPayload`, with `layout="node_records"` for the
  compatibility layout and `layout="split"` for the memory-oriented layout.
- a generic `load_sharded_search_checkpoint(...)` reader that reconstructs the
  typed `SearchRuntimeCheckpointPayload` from the C2b shards.

The C2c reader is an equivalence milestone. It validates the manifest and reads
the node-record shards in manifest order, but it still materializes the typed
payload graph before any runtime restore.

C2d1 adds an opt-in experimental `load_search_from_sharded_checkpoint(...)`
runtime restore path. It reads the same C2b layout, keeps slim per-node shells
for state handles and tree construction, and restores full node edge/evaluation
payloads one shard at a time. This avoids constructing a full typed
`SearchRuntimeCheckpointPayload`, but it is not the default path and should not
be treated as a measured memory win until profiled on large checkpoints.

C2e extends that incremental runtime reader to prefer the split layout when
`node_shells`, `state_payloads`, and `node_runtime` shards are present. Pass 1
reads only `state_payloads` and `node_shells`, builds the state payload store
and compact node shells, then pass 2 streams `node_runtime` shards for edges,
evaluation, and exploration-index state.

## Shard Boundaries

C2b shards large per-node data as complete node records:

- node structural payload,
- node state payload,
- linked child edges,
- node evaluation payload,
- exploration-index payload.

Future C2d work may split those records further:

- node structural payloads,
- node state payloads,
- linked child edges,
- node evaluation payloads.

Small or global data can remain in whole-file shards:

- metadata,
- root id,
- evaluator version,
- RNG state,
- selector state,
- latest tree expansions,
- future checkpoint build metrics.

Splitting node structure, state payloads, edges, and evaluation payloads lets a
future restore path process bounded raw graphs and drop each shard immediately.
It also lets the state payload store be built independently from node shell and
evaluation restoration.

## Restore Strategy

The split incremental restore should:

1. Read and validate `manifest.json`.
2. Read small metadata and global shards.
3. Build the state payload store incrementally from `state_payloads` shards.
4. Create checkpoint-backed state handles.
5. Stream `node_shells` shards to create runtime nodes.
6. Stream `node_runtime` shards after all nodes exist to restore edges,
   evaluation, and exploration-index state.
8. Restore selector state and latest expansions.
9. Drop each raw shard immediately after processing.
10. Run equivalence assertions in tests against the monolithic path.

This can share many existing restore helpers, but those helpers should gradually
move toward small field-access adapters rather than requiring a fully typed
`SearchRuntimeCheckpointPayload`.

## Equivalence Strategy

Future tests must export one runtime through both monolithic and sharded paths,
restore both, and compare:

- root id,
- node ids,
- branch count,
- parent and child links,
- state handle laziness and materialized state values,
- direct and backed-up node values,
- decision ordering,
- principal variation,
- branch frontier,
- backup runtime,
- exploration index data,
- selector behavior,
- latest expansions.

## Suggested PR Sequence

C2b should implement the generic sharded writer for small checkpoints and a
manifest-driven directory layout test. This is complete in the first-stage
node-record layout.

C2c implements a non-streaming sharded reader that reconstructs the typed
checkpoint payload for equivalence first. This does not reduce peak memory; it
proves the C2b layout can round-trip through a manifest-driven reader.

C2d1 adds an opt-in incremental runtime restore path for the first-stage C2b
node-record layout. It may still process node-record shards in multiple passes,
but it should not build all full typed node payloads at once.

C2d should continue making restore more incremental where useful: state payload
store, node shells, edges, and evaluation shards should be processed and dropped
in separate phases.

C2e adds the split writer and split incremental restore while preserving the
node-record writer/reader tests. Chipiron's opt-in sharded runtime checkpoint
path requests `layout="split"` for measurement; the monolithic default is
unchanged.
