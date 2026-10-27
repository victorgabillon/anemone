Python 3.14 migration, validated 27 October 2026

The ordinary checkout uses `migrate/python314-morpion`. The validated code and
configuration commit is `bb972947f1b2c44da65897de5c01bdc301bd5c67`.
CPython 3.14.10 runs with the GIL enabled and JIT disabled. Nothing was published,
pushed or merged, and no worktree was created.

The implementation foundation is `82b169faa47a38a25a5f51497afa442076598d02`,
six commits beyond public main `a07fb2fdd75e2020d9795d34ec1cf896eb86f3a0`.
It includes scientific certainty/effective-value propagation, bounded state
resolution, shared exact nonterminal scoring, ordered repeated continuation,
`37e48c40ceff430e373fd07bd00fcf41359bafda` streaming checkpoints, and the
subsequent state-free max comparison correction. Review those six commits
separately before reviewing the migration; they were not merged into main.
The pinned 37e48c4 wheel was verified against its supplied SHA-256 and all 290
packaged source/typing files. The later 82b169f wheel differs only in the two
state-free comparison production modules; checkpoint implementation is retained.

Native generic declarations replace nine legacy TypeVars. Search dynamics,
evaluation protocols and factories have actual state/evaluation bounds.
`StatelessDynamicsAdapter` now accepts one state parameter and exposes Valanga's
real `BranchKey` contract, removing two unsound return suppressions. Recursive
defaults remain available. Positive variance/default/bound fixtures pass both
checkers; each rejects all seven deliberate invalid uses. Runtime annotation
bindings needed by public factories and Parsley remain available. The optional
Torch evaluator now has a typed structural backend, declared slots, current
`Value.score` adaptation and a correctly typed module-scripting boundary.

The migration removes 166 production future-annotation imports and obsolete
compatibility scaffolding. Production Python physical lines decrease from
42,385 to 42,185; `Any` mentions remain 875, and type-ignore directives fall from
seven to five. Native max-heaps were considered and rejected: the current
negative-priority tuple expresses existing tie/stale-entry ordering more simply.
Checkpoint format 3 and the Zstandard codec are unchanged.

Full tox passes: **1,597 tests, no skips**, Ruff check/format, strict mypy,
strict Pyright, Pylint 10/10, positive/negative contracts, wheel/sdist builds
and strict Twine validation. All original 1,583 test identities remain selected.
Sphinx builds with warnings treated as errors. Workflows pass actionlint 1.7.12;
remote GitHub Actions has not run. Existing analyzer-debt baseline files remain
historical evidence; CI now requires zero diagnostics directly.

Fixed Morpion integration matches Python 3.13 exactly for ordered DAGs,
direct/tree/effective values, FORCED/TERMINAL/ESTIMATE distinctions, all compressed
checkpoint files, bounded resolver capacity three, independent rollout RNG and
12 continued Growth calls with a second restore halfway through. Installed
Anemone and Coral also complete a scripted-MLP neural search/save/restore and
eight subsequent Growth calls with exact continuation.

Matched CPU measurements at 5,013 nodes give median Growth times 10.173 s for
original source on 3.14 and 10.177 s for migrated source with the same foundation
wheels. The full requested foundation stack measures 10.468 s; modest timing and
RSS differences are reported without claiming a general interpreter speedup.
All 12 final search runs match every checkpoint byte and selection trace.

Existing edits are preserved unchanged in
`preserve/pre-python314-20261027`, commit
`c5612e6fc5c83c24cfdf79bb4cfb88b5e1ef7790`, outside migration ancestry.
Alternating policy/counter functionality and parser-compatible string arguments
are already incorporated. Saved default-policy changes, unconditional inverse
probability diagnostics and the `deterministic_index` display label remain
separate review candidates. Their weaker duplicate implementations are
superseded; the preservation commit remains intact.

The tested preview is `algorhino-anemone==0.2.27.dev0`, wheel SHA-256
`98738440bc8564a73cd3bd086f8b942b030477363f042ce7e5b6910bc90f4c96`.
All 290 installed/package source files match the validated Git commit.
Exact foundation commit-built wheels retain their old version labels and are
supplied explicitly; plain index resolution does not identify this stack.

Recommend final distribution **0.3.0** for the changed Python floor and public
generic contracts. Before publishing, align dependencies to published
`valanga==0.3.0`, test extras `atomheart==0.3.0` and `parsley-coco==0.2.0`, and the
migrated Coral optional dependency; rebuild and rerun release validation with
those identities. Their published foundation wheels have byte-identical
packaged source to the requested commit-built wheels, verified separately.
The scientific stack needs review/integration and remote CI before release.

Full local evidence, immutable tested wheels, source hashes, performance ranges,
preservation classifications and review commands are in
`/home/pompote/oldata/victor/python314-anemone-coral-0shyf5le/REPORT.md`.
