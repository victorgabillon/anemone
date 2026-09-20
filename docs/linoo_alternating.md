# Optional alternating Linoo policy

`LinooArgs` and `Linoo` keep `inverse_depth` as their public default. An application
can explicitly select `depth_selection_policy="alternating_by_step"`.

The selector counts successful depth choices, starting at zero. It computes the
next count before choosing a depth and stores it immediately after that choice:

- Odd counts (1, 3, ...) use the existing inverse-depth sampler with weight
  `1 / (depth + 1)` and its existing random generator.
- Even counts (2, 4, ...) use the existing deterministic minimum of
  `(opened_count * (depth + 1), depth)` without a random draw for depth choice.
- Within-depth Zipf node ranking and its random draws are unchanged. Consequently
  an even **selection** can still consume randomness for node ranking.

This is a selector-instance depth-selection count, not the global tree-growth
step. Calls bypassing the selector do not count. Continuing a search preserves
the count. Cache invalidation/rebuild preserves it, including a checkpoint saved
after invalidation. A new independent search creates a new selector with count
zero; reusing an existing selector explicitly continues its count.

## Checkpoints

`LinooSelectorCheckpointPayload.selection_step_count` is durable state even when
the optional cached frontier is stale and must be rebuilt. Typed payload restore,
sharded decoding, and incremental streaming restoration preserve it. The shared
mapping decoder serves both sharded restoration paths. Composed selectors now
export their existing shared RNG state; JSON restoration reconstructs the RNG
state vector tuple. Neither change adds random draws to live selection.

The field is additive with a zero default, so payload version 1 / checkpoint
format version 2 remain unchanged. Existing decoders ignore unrecognized optional
fields; neither published legacy policy uses the count to choose a depth. Old
payloads without a count remain readable and start at count zero: their next
alternating selection uses step 1. Exact prior alternating parity cannot be
recovered from such a file. Alternating was never a previously published policy,
so there was no published alternating-checkpoint guarantee to preserve. Invalid
negative, boolean, or noninteger counters are rejected.

Anemone exports generic monolithic JSON but domain applications supply their
typed monolithic decoder. Chipiron's actual monolithic normalization was checked
against the new payload at both parities; its full runner boundary is validated
in Chipiron. Anemone's tests exercise typed runtime restore, sharded payload
decode/restore, and streaming restore of both node-record and split layouts.

## Diagnostics and prototype provenance

Reports add configured/effective policy, selector step, and odd/even parity. For
alternating, candidate index and inverse-depth reference weight/probability are
available on both kinds of step. On an even step these probabilities describe the
inverse-depth alternative, not the deterministic step's sampling distribution.
The standalone index policy retains its existing absent stochastic diagnostics.
The existing `index` table heading is unchanged; new optional report fields are
appended to preserve existing positional field order.

This feature was reconstructed from the intended six-file local prototype
identified in the cross-repository audit of 2026-09-20, then integrated on the
reconciled 0.2.23 main. The original dirty workspace remains unchanged. Eight
selections over each of ten seeded frontiers matched the preserved prototype for
node/depth choice, subpolicy, step/parity, indices, weights/probabilities and RNG
state. Intentional differences are:

- The global default stays inverse-depth, and the 0.2.23 parser/TypeIs validation
  is retained.
- Sharded counter loss, composed-selector RNG omission, and JSON RNG decoding
  are corrected; invalid counters are rejected.
- Cache invalidation or stale-cache restore cannot discard a valid saved count.
- The prototype's cosmetic table rename and standalone deterministic-policy
  probability-display change are excluded to preserve released diagnostics.
- Direct alternating dispatch rejects a missing or nonpositive step instead of
  interpreting zero/negative values as valid depth selections.

No node-ranking, opening, recommendation, budget, evaluation, scheduling or seed
policy changes are part of this feature.
