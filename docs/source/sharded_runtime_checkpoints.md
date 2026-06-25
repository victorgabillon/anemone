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

## Non-goals for C2a

- No default behavior change.
- No migration from existing monolithic checkpoints.
- No full sharded writer or reader.
- No streaming JSON parser dependency.
- No Morpion-specific optimization in Anemone.
- No claimed memory savings from the skeleton alone.

## Proposed Layout

One sharded checkpoint is a directory rooted by `manifest.json`:

```text
checkpoint_sharded/
  manifest.json
  metadata.json.zst
  tree_nodes/
    nodes_000000.jsonl.zst
    nodes_000001.jsonl.zst
  state_payloads/
    state_payloads_000000.jsonl.zst
  linked_children/
    linked_children_000000.jsonl.zst
  node_evaluations/
    evaluations_000000.jsonl.zst
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

The C2a skeleton provides:

- `ShardedCheckpointManifest`,
- `ShardedCheckpointShardRef`,
- manifest JSON conversion,
- manifest read/write helpers,
- relative shard path validation.

Full runtime checkpoint read/write functions are deliberately placeholders until
the writer and restore plan have tests.

## Shard Boundaries

Large per-node data should be sharded:

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

A future sharded restore should:

1. Read and validate `manifest.json`.
2. Read small metadata and global shards.
3. Build the state payload store incrementally from `state_payloads` shards.
4. Create checkpoint-backed state handles.
5. Stream `tree_nodes` shards to create runtime nodes.
6. Stream `linked_children` shards after all nodes exist.
7. Stream `node_evaluations` shards to restore node runtime state.
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
manifest-driven directory layout test.

C2c should implement a non-streaming sharded reader that loads shards in bounded
groups but reuses existing restore semantics, then add equivalence tests.

C2d should make the reader truly incremental where useful: state payload store,
node shells, edges, and evaluation shards should be processed and dropped in
separate phases.

Only after those pass should Chipiron add an opt-in runtime checkpoint format
switch for large Morpion runs.
