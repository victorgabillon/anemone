"""Torch-based value evaluator for efficient batch evaluations."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from valanga import State
from valanga.evaluations import Certainty, EvalItem, Value

from anemone.node_evaluation.direct.protocols import (
    MasterStateValueEvaluator,
    OverEventDetector,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from torch import Tensor
    from torch.nn import Module


class _OutputValueConverter(Protocol):
    def to_content_evaluation(self, output_nn: Tensor, state: State) -> Value: ...


class _TorchStateEvaluator(Protocol):
    @property
    def net(self) -> Module: ...

    @property
    def content_to_input_convert(self) -> Callable[[State], Tensor]: ...

    @property
    def output_and_value_converter(self) -> _OutputValueConverter: ...


class TorchDependencyError(ModuleNotFoundError):
    """Raised when torch is required but not installed."""

    def __init__(self) -> None:
        """Initialize the error for missing torch dependencies."""
        super().__init__(
            "TorchMasterNNStateEvaluator requires 'torch'. "
            "Install the optional torch dependencies."
        )


@dataclass(slots=True)
class TorchMasterNNStateEvaluator(MasterStateValueEvaluator):
    """Torch-backed value evaluator that supports efficient batch evaluation.

    This lives in an optional module so anemone core has no torch dependency.
    """

    over: OverEventDetector
    state_evaluator: _TorchStateEvaluator
    device: str = "cpu"
    script: bool = True
    _torch: ModuleType = field(init=False, repr=False)
    _model: Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the torch model and validate dependencies."""
        try:
            import torch  # pylint: disable=import-outside-toplevel
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

    def evaluate(self, state: State) -> Value:
        """Evaluate one state into an estimate Value."""
        return Value(
            score=float(self.value_white(state)),
            certainty=Certainty.ESTIMATE,
            over_event=None,
        )

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
            values.append(float(state_eval.score))

        return values

    def evaluate_batch_items[ItemStateT: State](
        self, items: Sequence[EvalItem[ItemStateT]]
    ) -> list[Value]:
        """Evaluate a batch of items into estimate Values."""
        scores = self.value_white_batch_items(items)
        return [
            Value(score=float(score), certainty=Certainty.ESTIMATE, over_event=None)
            for score in scores
        ]


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
