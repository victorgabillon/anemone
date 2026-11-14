# Changelog

## 0.4.0

### Added

- Native prepared evaluation through `anemone.prepared_evaluator`: immutable
  source freezing, detached evaluator preparation, and staged installation into
  an exclusively owned runtime restored from the same checkpoint.
- Canonical null indexes and `MIN_GLOBAL_CHANGE` indexes in the supported
  single-agent max/proof, noop/Linoo recipe. Admission binds source, evaluator,
  codec, owner and recipe identities; unsupported or inconsistent inputs refuse.
- Generic owned-inline codec capabilities, with explicit reconstruction and
  ownership requirements independent of any particular game.
- Complete typed materialization of both `node_records` and `split` checkpoints
  through `load_sharded_search_checkpoint`, with optional explicit
  `ShardedCheckpointReadLimits` for input sizes, work deadlines and cancellation.
  Prepared consumers must supply limits.

### Preserved behavior

- Ordinary node reevaluation, partial refresh and incremental value propagation
  remain available. Full evaluator refresh uses one shared canonical kernel.
- `SearchRuntimeCheckpointPayload` remains the canonical runtime schema at
  version 3; sharded manifests remain version 1. Streaming restoration remains
  available without routing through full materialization.
- A runtime invalidated by a partially applied prepared installation refuses
  further searching and checkpoint publication; discard it and restore again.
- Python 3.14 or higher and `valanga==0.3.0` remain required. No application
  smoothing policy, automatic activation or scientific search recipe is added.

### Packaging

- Declare the Python `graphviz` package as a runtime dependency because public
  imports require it. The `debug` extra remains accepted; the system Graphviz
  executable is still needed only for rendering.

### Validation and scope

- Independent architecture review and 721 affected tests passed in each of two
  dependency environments, with no failures or skips.
- Installed-package checks cover public imports, bounded SPLIT materialization,
  both supported index recipes, prepared installation and exact checkpoint
  parity after subsequent Growth.
- This release adds a bounded capability; it does not establish a production
  performance improvement or enable it in Chipiron. Downstream dependency
  changes and activation remain separate integration decisions.
