# Node evaluation

Node evaluation is split into two layers:

- **Common** (`common/`): canonical `Value` semantics and the shared
  node-evaluation protocol.
- **Direct evaluation** (`direct/`): evaluates a single game state and exposes
  `NodeDirectEvaluator` along with the `MasterStateEvaluator` protocol.
- **Tree evaluation** (`tree/`): aggregates child evaluations and computes
  backed-up values for the single-agent and adversarial search families.

Factories in these modules are used by the main `factory.py` entry points to
build the evaluation pipeline for the tree-and-value search.
