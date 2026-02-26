"""Reusable backup operators for objective-driven search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class BackupOp:
    """Base callable backup operator."""

    def __call__(self, values: Sequence[float]) -> float:
        """Compute one backed-up value from a sequence of child values."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class MaxBackup(BackupOp):
    """Back up by taking the maximum child value."""

    def __call__(self, values: Sequence[float]) -> float:
        """Return the max value."""
        return max(values)


@dataclass(frozen=True, slots=True)
class MinBackup(BackupOp):
    """Back up by taking the minimum child value."""

    def __call__(self, values: Sequence[float]) -> float:
        """Return the min value."""
        return min(values)


@dataclass(frozen=True, slots=True)
class MeanBackup(BackupOp):
    """Back up by taking arithmetic mean."""

    def __call__(self, values: Sequence[float]) -> float:
        """Return the mean value."""
        return sum(values) / len(values)


@dataclass(frozen=True, slots=True)
class SoftmaxBackup(BackupOp):
    """Back up using log-sum-exp scaled by ``beta``."""

    beta: float = 1.0

    def __call__(self, values: Sequence[float]) -> float:
        """Return stable softmax-style backup score."""
        m = max(values)
        if self.beta == 0:
            return sum(values) / len(values)
        s = sum(math.exp(self.beta * (v - m)) for v in values)
        return m + math.log(s) / self.beta
