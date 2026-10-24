# Bounded checkpoint decoded-state retention

Task A replaces the decode cache inside the existing `CheckpointStateResolver`
with one `OrderedDict` LRU. It retains the original payload store, decoder,
anchor/delta policy and active-chain cycle detection.

Both public loaders, `load_search_from_checkpoint_payload` and
`load_search_from_sharded_checkpoint`, accept `state_cache_capacity` (default 256).
The same constructor argument is available when constructing the resolver directly.
It is a non-negative integer, fixed for the resolver's lifetime; zero disables
retention. The default is also exported as `anemone.checkpoints.DEFAULT_STATE_CACHE_CAPACITY`.
No checkpoint schema/version, payload bytes or persisted policy changes.

Every hit moves a resident state to the most recent end. Each successful decode
inserts its result and evicts the least recently used entry if necessary, before
returning. Thus every completed resolution leaves at most the configured capacity
resident. Cache capacity does not depend on the number of visited nodes. There is
no per-decoded-node metadata outside this LRU. The existing resolving-ID set tracks
only the active call chain and clears on success or exception.

A child's reconstruction holds its concrete parent on the Python call stack,
independently of cache membership. Reentrant eviction cannot invalidate that
reference. Existing external references remain valid; thin checkpoint handles
continue resolving from the original payload authority after eviction. Repeated
`get()` calls return the identical object while it remains cached; after eviction
or with capacity zero they guarantee equivalent reconstructed state, not Python
object identity. Codecs must retain their existing contract of reconstructing a
child without mutating its parent.

This bounds only additional cached decoded states. Payload indices, live graph,
selector and PV storage remain population-sized. Active reconstruction chains and
individual state sizes can grow. Recursive chain depth retains the existing Python
recursion limit; tests exercise 180 deltas and reentrant decoding. A state still
referenced by a caller is intentionally not freed by cache eviction.

Fresh real Morpion graphs have 1,003/5,013 nodes, 1,114/5,931 edges, depths 20/62 and
106/759 shared nodes. Paired baseline/bounded runs use the exact same newly generated
checkpoint for each size, window 64 and raw predictions 42 then 7. Streaming parity
checks between passes cover canonical states, Values, independent certainties,
effective source, PV, decision order, all graph links, completeness and versions.
RNG and selector parity are checked around each refresh. The full resource report
records the CPU/reconstruction cost of bounded retention; warm unbounded caches
avoid decoding work which a finite cache must repeat.

The default 256 was chosen after comparing capacities 64, 256 and 1,024. On the 5k
fixture, 256 reduced peak RSS from about 140 MiB to 76 MiB with about 10% more first
refresh time; 1,024 retained about 100 MiB and saved few extra decodes. These are
single synthetic CPU trials with a controlled evaluator, not neural activation or
constant-total-memory claims. Capacity is explicit so applications can choose a
larger CPU/memory tradeoff after measuring their workload.
