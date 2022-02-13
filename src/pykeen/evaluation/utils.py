# -*- coding: utf-8 -*-

"""Utilities for evaluation."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import rexmex.utils

__all__ = [
    "MetricAnnotation",
    "ValueRange",
]


@dataclass
class ValueRange:
    """A value range description."""

    #: the lower bound
    lower: Optional[float] = None

    #: whether the lower bound is inclusive
    lower_inclusive: bool = False

    #: the upper bound
    upper: Optional[float] = None

    #: whether the upper bound is inclusive
    upper_inclusive: bool = False

    def __contains__(self, x: float) -> bool:
        """Test whether a value is contained in the value range."""
        if self.lower is not None:
            if x < self.lower:
                return False
            if not self.lower_inclusive and x == self.lower:
                return False
        if self.upper is not None:
            if x > self.upper:
                return False
            if not self.upper_inclusive and x == self.upper:
                return False
        return True

    def notate(self) -> str:
        """Get the math notation for the range of this metric."""
        left = "(" if self.lower is None or not self.lower_inclusive else "["
        right = ")" if self.upper is None or not self.upper_inclusive else "]"
        return f"{left}{self._coerce(self.lower)}, {self._coerce(self.upper)}{right}"

    @staticmethod
    def _coerce(n: Optional[float]) -> str:
        if n is None:
            return "inf"  # ∞
        if isinstance(n, int):
            return str(n)
        if n.is_integer():
            return str(int(n))
        return str(n)


@dataclass
class MetricAnnotation:
    """Metadata about a classifier function."""

    name: str
    increasing: bool
    value_range: ValueRange
    description: str
    link: str

    binarize: Optional[bool] = None
    func: Optional[Callable[[np.array, np.array], float]] = None

    def __post_init__(self):
        """Prepare the function by binarizing it if annotated."""
        if self.binarize:
            if self.func is None:
                raise ValueError
            self.func = rexmex.utils.binarize(self.func)

    def score(self, y_true, y_score) -> float:
        """Run the scoring function."""
        if self.func is None:
            raise ValueError
        return self.func(y_true, y_score)
