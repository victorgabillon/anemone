# Typed sharded-checkpoint materialization

`anemone.checkpoints.load_sharded_search_checkpoint` returns the existing complete
`SearchRuntimeCheckpointPayload` from either native layout:

| Input | Materialization |
| --- | --- |
| `node_records`, JSONL or JSONL+zstd | Supported, including existing calls without limits |
| `split`: `node_shells` + `state_payloads` + `node_runtime` | Supported, joined by exact integer node ID |
| Mixed, incomplete, duplicate or conflicting families | Refused |

Metadata and optional selector/latest-expansion singleton shards accept JSON or
JSON+zstd. Shell/manifest order and all stored branch/parent sequences are retained;
state and runtime rows may arrive in different orders. State-codec ancestry is
independent of graph ancestry. Literal `None` indexes are retained alongside the
existing non-null index payloads. This reader does not select an index recipe.

Split ingestion requires complete native fields, including optional fields stored
as explicit nulls. It rejects contradictory summaries, missing/extra IDs, duplicate
JSON keys, metadata/manifest count conflicts and corrupt/truncated compression.
Stored selector and proof data is decoded into native dataclasses without repair;
prepared admission remains responsible for scientific validity. No domain codec,
evaluator, selector or runnable search is constructed, and no checkpoint is written.

## Explicit bounds

```python
from time import monotonic
from anemone.checkpoints import (
    ShardedCheckpointReadLimits,
    load_sharded_search_checkpoint,
)

limits = ShardedCheckpointReadLimits(
    deadline_monotonic=monotonic() + 10,
    max_nodes=512,
    max_branches=4096,
    max_shards=1024,
    max_compressed_bytes=16 * 1024 * 1024,
    max_uncompressed_bytes=32 * 1024 * 1024,
    max_record_bytes=4 * 1024 * 1024,
)
payload = load_sharded_search_checkpoint(checkpoint_dir, limits=limits)
```

These are illustrative small-fixture limits, not production defaults or enlarged
production caps. `ShardedCheckpointReadLimits` is frozen. All maxima are positive
integers; `cancelled` is an optional zero-argument callback. One absolute monotonic
deadline covers manifest reading, all shard families, joining and validation.
Prepared consumers must pass explicit limits at this ingestion boundary, then
pass appropriate `PreparationLimits` to freeze/prepare/stage. The ordinary reader's
`limits=None` contract remains unbounded.

`max_compressed_bytes` counts actual file bytes, including the plain manifest and
any uncompressed files. `max_uncompressed_bytes` counts all decoded bytes, including
whitespace and singleton records. `max_record_bytes` applies before JSON parsing to
each JSONL row (excluding LF), each singleton document, and the whole manifest.
The shard count covers every manifest entry, including singleton shards. Node and
branch bounds cover actual complete nodes and linked edges, as well as declared
totals. Actual file bytes, decoded bytes and per-shard record counts are checked;
optional byte/digest declarations are checked for consistency and are never a
substitute for measuring input or authenticating the publication.

Input and decompressed chunks are at most 64 KiB, with smaller reads near aggregate
limits. Python 3.14's standard-library zstd decoder caps returned output and detects
frame completion, including concatenated frames. For bounded reads its native
window ceiling is the next power of two above the uncompressed-byte bound (minimum
1 KiB, maximum 2 GiB). Path containment checks remain in force. Callers must pin an
immutable authenticated directory and retain hard process time/RSS/output limits;
cooperative checks cannot interrupt arbitrary Python cancellation callbacks or a
single native operation. Exceptions return no partial usable payload.

## Storage and subsequent preparation

Materialization requires O(N + E + P) complete typed storage, plus ID maps and
bounded parsing buffers. Split state/runtime rows are converted immediately to
native typed objects; joining consumes them in shell order. There is no second
complete raw JSON tree. The current row, JSON parsing allocations and a few last
row references still coexist with typed storage. Byte limits are not RSS limits.

Freeze serializes the complete result and reconstructs a detached graph/codec
arena. Preparation reconstructs another arena, and staging coexists with the cold
owner and staged derived objects. Input materialization does not remove those
costs, full catch-up, source matching or publication. Small generated observations
are diagnostics; production memory and activation cost still require qualification.

`tests/test_checkpoint_materialize.py` covers both layouts/encodings/index recipes,
complete typed fields, limits and negative inputs, then consumes the original
persisted split source through freeze/prepare/install against independently restored
owners, with exact checkpoint/export and nonzero Growth/RNG correspondence.
