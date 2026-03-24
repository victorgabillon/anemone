# Profiling Foundation

This document describes the first standalone profiling foundation for Anemone.

## Purpose

PR1 adds a small profiling shell that can:

- run a named scenario
- measure total wall-clock time
- capture basic execution metadata
- persist a stable JSON artifact for each run

The profiling foundation is intentionally separate from the core search engine.

## Constraints

This first version does not modify core search semantics and does not add timing
or profiler hooks inside the search loop.

In particular, PR1 does not touch:

- `TreeExploration.step()`
- `AlgorithmNodeTreeManager`

## What PR1 Includes

- `anemone.profiling` package under `src/anemone/profiling/`
- stable `RunResult` JSON schema with `schema_version = "1"`
- run directory helpers
- a standalone scenario runner
- a minimal CLI
- a tiny real `smoke` scenario using public Anemone APIs
- lazy scenario loading so package import stays lightweight

## What PR1 Does Not Include

- external profiler integrations such as py-spy, pyinstrument, or viztracer
- wrapper-based evaluator or selector timing
- benchmark matrices
- HTML reports, charts, or GUI tooling

## Running It

Preferred CLI:

```bash
python -m anemone.profiling.cli list-scenarios
python -m anemone.profiling.cli run --scenario smoke --output-dir profiling_runs
```

Convenience runner entrypoint:

```bash
python -m anemone.profiling.runner --scenario smoke --output-dir profiling_runs
```

## Output Layout

Each run creates a timestamped folder under the selected base directory:

```text
profiling_runs/
  2026-03-24T14-32-10_smoke/
    run.json
```

If a run id collides, the storage helper appends a deterministic suffix such as
`_2`, `_3`, and so on.

The `run.json` artifact includes scenario metadata, execution metadata, top-level
wall time, run status, and placeholder artifact references for future tooling.
