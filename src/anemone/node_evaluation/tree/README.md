# Tree-based evaluation

Tree evaluation aggregates direct evaluations across children to compute
backed-up values for search nodes.

## Key pieces

- `single_agent/node_single_agent_evaluation.py`: `NodeSingleAgentEvaluation`
  protocol for the single-agent family.
- `single_agent/node_max_evaluation.py`: Single-agent max implementation that
  tracks best branches and canonical values.
- `adversarial/node_adversarial_evaluation.py`: `NodeAdversarialEvaluation`
  protocol for adversarial backup semantics.
- `adversarial/node_minmax_evaluation.py`: Minmax implementation that tracks
  best branches, per-branch ordering, and terminal status.
- `factory.py`: Factories for constructing tree-evaluation implementations,
  including the default minimax evaluator.
