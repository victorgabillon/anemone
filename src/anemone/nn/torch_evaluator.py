"""Torch-based MasterStateEvaluator for efficient batch evaluations."""
# pyright: reportMissingImports=false

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from valanga import State
from valanga.evaluations import EvalItem

from anemone.node_evaluation.node_direct_evaluation.node_direct_evaluator import (
    MasterStateEvaluator,
    OverEventDetector,
)

if TYPE_CHECKING:
    from coral.neural_networks.nn_content_evaluator import NNContentEvaluator
    from torch import Tensor


class TorchDependencyError(ModuleNotFoundError):
    """Raised when torch is required but not installed."""

    def __init__(self) -> None:
        """Initialize the error for missing torch dependencies."""
        super().__init__(
            "TorchMasterNNStateEvaluator requires 'torch'. "
            "Install the optional torch dependencies."
        )


@dataclass(slots=True)
class TorchMasterNNStateEvaluator(MasterStateEvaluator):
    """Torch-backed MasterStateEvaluator that supports efficient batch evaluation.

    This lives in an optional module so anemone core has no torch dependency.
    """

    over: OverEventDetector
    state_evaluator: "NNContentEvaluator"
    device: str = "cpu"
    script: bool = True

    def __post_init__(self) -> None:
        """Initialize the torch model and validate dependencies."""
        try:
            import torch
        except ModuleNotFoundError as e:
            raise TorchDependencyError from e

        self._torch = torch

        model = self.state_evaluator.net
        self._model = torch.jit.script(model) if self.script else model
        self._model.eval()

    def value_white(self, state: State) -> float:
        """Evaluate a single state by delegating to the batch path."""
        # Slow path: evaluate a single state by wrapping it as an EvalItem.
        return self.value_white_batch_items([_SingleEvalItem(state)])[0]

    def value_white_batch_items[ItemStateT: State](
        self, items: Sequence[EvalItem[ItemStateT]]
    ) -> list[float]:
        """Evaluate a batch of items with torch and return white values."""
        torch = self._torch

        xs: list[Tensor] = []
        states: list[ItemStateT] = []

        for it in items:
            st = it.state
            states.append(st)

            # Prefer precomputed representation when available
            if it.state_representation is not None:
                raw = it.state_representation.get_evaluator_input(state=st)
            else:
                raw = self.state_evaluator.content_to_input_convert(st)

            xs.append(torch.as_tensor(raw).to(self.device))

        x_batch = torch.stack(xs, dim=0)

        with torch.no_grad():
            out = self._model(x_batch)

        converter = self.state_evaluator.output_and_value_converter

        values: list[float] = []
        for i, st in enumerate(states):
            state_eval = converter.to_content_evaluation(output_nn=out[i], state=st)
            vw = state_eval.value_white
            assert vw is not None
            values.append(float(vw))

        return values


class _SingleEvalItem:
    """Small adapter so we can call batch method from value_white."""

    def __init__(self, state: State) -> None:
        """Initialize the EvalItem with the given state."""
        self._state = state

    @property
    def state(self) -> State:
        """The state to evaluate."""
        return self._state

    @property
    def state_representation(self) -> None:
        """No precomputed representation."""
        return None
