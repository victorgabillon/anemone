# Profiling Foundation

This document describes the standalone profiling foundation for Anemone.

## Purpose

The profiling package can:

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

## What Exists Today

- `anemone.profiling` package under `src/anemone/profiling/`
- stable `RunResult` JSON schema with `schema_version = "1"`
- run directory helpers
- a standalone scenario runner
- a minimal CLI
- a tiny real `smoke` scenario using public Anemone APIs
- lazy scenario loading so package import stays lightweight
- optional external profiler modes: `none`, `cprofile`, `pyinstrument`
- wrapper-based component timing for evaluator and dynamics
- `component_summary.json` artifact output when enabled
- `cprofile.pstats` and `cprofile_top.txt` artifacts for `cprofile`
- `pyinstrument.txt` artifact when `pyinstrument` is requested and installed

## What Is Still Out Of Scope

- process-level profilers such as py-spy
- viztracer integration
- benchmark matrices
- HTML reports, charts, or GUI tooling

## Running It

Preferred CLI:

```bash
python -m anemone.profiling.cli list-scenarios
python -m anemone.profiling.cli run --scenario smoke --output-dir profiling_runs
python -m anemone.profiling.cli run --scenario smoke --output-dir profiling_runs --profiler cprofile --component-summary
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
    component_summary.json
    cprofile.pstats
    cprofile_top.txt
```

If a run id collides, the storage helper appends a deterministic suffix such as
`_2`, `_3`, and so on.

The `run.json` artifact includes scenario metadata, execution metadata, top-level
wall time, run status, and placeholder artifact references for future tooling.
The recorded wall time tracks scenario execution itself and intentionally
excludes profiler artifact writing or profiler post-processing.

When component summaries are enabled, the summary approximates framework
overhead as:

```text
residual_framework_wall_time_seconds
    = total_run_wall_time_seconds - wrapped_component_wall_times
```

This residual is useful, but intentionally approximate.
