# Shared nonterminal numeric transformation

Task B extracts the original Growth expression into the public scalar helper:

```python
from anemone.node_evaluation.direct import scale_nonterminal_score

score = scale_nonterminal_score(raw_score, tree_depth)
```

`tree_depth` is checkpoint-relative depth, exactly the value used by the existing
Anemone node evaluator. An endgame checkpoint root at Morpion move 34 has depth
zero, not 34. The helper takes/returns a scalar and does not create or change a
Value's certainty. Terminal scores must bypass it.

The function is exactly `(1 / DISCOUNT) ** tree_depth * raw_score` with
`DISCOUNT = 0.99999999`. There is no changed constant, cast, rounding, reassociation,
finite-value rejection, discount policy or objective. The old
`NodeDirectEvaluator.process_evalution_not_over()` delegates to it; the existing
`node_direct_evaluator.DISCOUNT` import remains available from the single constant
definition. Existing certificate validation and terminal handling remain intact.

The new test module compares packed IEEE-754 bytes against the frozen original
expression over 65 combinations: depths 0, 1, 62, one million and one billion;
positive/negative scores, signed zeros, largest finite magnitudes, subnormal,
infinities and NaNs including a specified quiet-NaN payload. Additional cases check
overflow, equal scores and independent Value certainty. The actual one-action
Morpion endgame retains direct ESTIMATE(42), tree/effective FORCED(35) and terminal
TERMINAL(35); its root is at played move 34 but tree depth zero. All 69 tests pass.

The operation is available for a future Chipiron integration; this task changes no
Chipiron production source. The model-patch adapter should pass the selected
checkpoint node's relative search depth and apply the helper only to nonterminal
raw model scores, preserving its estimated certainty. Certainty must come from the
selected Value, never from an effective exact flag elsewhere on the node.
Existing raw patch rows are not silently reinterpreted. The future authority and
proposal contracts still need to bind this numeric policy and reject incompatible
pending proposals. Dataset targets and historical checkpoints are unchanged.
