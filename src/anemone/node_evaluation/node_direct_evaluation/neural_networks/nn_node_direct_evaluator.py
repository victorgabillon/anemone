"""
This module contains the implementation of the `NNNodeEvaluator` class, which is a generic neural network class for board evaluation.
"""

from typing import TYPE_CHECKING, Protocol

import torch

from typing import Sequence


from anemone.nodes.algorithm_node.algorithm_node import AlgorithmNode
from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import NodeBatchValueEvaluator  
from valanga import FloatyStateEvaluation

from anemone.nodes.algorithm_node.algorithm_node import (
    AlgorithmNode,
)

from coral.neural_networks.nn_content_evaluator import NNContentEvaluator
from ..node_direct_evaluator import MasterStateEvaluator

if TYPE_CHECKING:
    from valanga import FloatyStateEvaluation


class MasterNNStateEvaluator(MasterStateEvaluator, Protocol):
    state_evaluator: NNContentEvaluator



class TorchNodeBatchDirectEvaluator(NodeBatchValueEvaluator):
    def __init__(
        self,
        master_nn_state_evaluator: MasterNNStateEvaluator,
        device: str = "cpu",
        script: bool = True,
    ) -> None:
        self.device = device
        self.nn_state_evaluator = master_nn_state_evaluator.state_evaluator

        model = self.nn_state_evaluator.net
        self.model = torch.jit.script(model) if script else model
        self.model.eval()

    def value_white_batch_from_nodes(self, nodes: Sequence[AlgorithmNode]) -> list[float]:
        # Build batch input
        xs: list[torch.Tensor] = []
        for n in nodes:
            raw = (
                n.state_representation.get_evaluator_input(state=n.state)
                if n.state_representation is not None
                else self.nn_state_evaluator.content_to_input_convert(n.state)
            )
            x = torch.as_tensor(raw).to(self.device)
            xs.append(x)

        x_batch = torch.stack(xs, dim=0)

        # Forward
        with torch.no_grad():
            output_layer = self.model(x_batch)

        # Decode to value_white using context (no state passed into converter)
        values: list[float] = []
        converter = self.nn_state_evaluator.output_and_value_converter
        for i, n in enumerate(nodes):
            state_eval: FloatyStateEvaluation = converter.to_content_evaluation(
                output_nn=output_layer[i],
                state=n.state,
            )
            # Best: make value_white non-optional and delete this assert.
            value_white:float|None = state_eval.value_white
            assert value_white is not None
            values.append(value_white)

        return values
