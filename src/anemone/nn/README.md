# Neural evaluators

This folder contains optional integrations that rely on extra dependencies.

- `torch_evaluator.py` implements `TorchMasterNNStateEvaluator`, a
  torch-backed `MasterStateEvaluator` that batches evaluations for speed.

Install the optional `nn` extra to use these integrations.
