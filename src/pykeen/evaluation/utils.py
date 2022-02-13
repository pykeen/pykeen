# -*- coding: utf-8 -*-

"""Utilities for evaluation."""

from dataclasses import dataclass
from typing import Callable, MutableMapping, Optional

import numpy as np
import rexmex.utils

__all__ = [
    "MetricAnnotation",
    "MetricAnnotator",
    "construct_indicator",
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


def interval(self) -> str:
    """Get the math notation for the range of this metric."""
    return self.value_range.notate()


class MetricAnnotator:
    """A class for annotating metric functions."""

    metrics: MutableMapping[str, MetricAnnotation]

    def __init__(self):
        self.metrics = {}

    def higher(self, func, **kwargs):
        """Annotate a function where higher values are better."""
        return self.add(func, increasing=True, **kwargs)

    def lower(self, func, **kwargs):
        """Annotate a function where lower values are better."""
        return self.add(func, increasing=False, **kwargs)

    def add(
        self,
        func,
        *,
        increasing: bool,
        description: str,
        link: str,
        name: Optional[str] = None,
        lower: Optional[float] = 0.0,
        lower_inclusive: bool = True,
        upper: Optional[float] = 1.0,
        upper_inclusive: bool = True,
        binarize: bool = False,
    ):
        """Annotate a function."""
        self.metrics[func] = MetricAnnotation(
            func=func,
            binarize=binarize,
            name=name or func.__name__.replace("_", " ").title(),
            value_range=ValueRange(
                lower=lower,
                lower_inclusive=lower_inclusive,
                upper=upper,
                upper_inclusive=upper_inclusive,
            ),
            increasing=increasing,
            description=description,
            link=link,
        )


def construct_indicator(*, y_score: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Construct binary indicators from a list of scores.

    If there are $n$ positively labeled entries in ``y_true``, this function
    assigns the top $n$ highest scores in ``y_score`` as positive and remainder
    as negative.

    :param y_score:
        A 1-D array of the score values
    :param y_true:
        A 1-D array of binary values (1 and 0)
    :return:
        A 1-D array of indicator values

    .. seealso::

        This implementation was inspired by
        https://github.com/xptree/NetMF/blob/77286b826c4af149055237cef65e2a500e15631a/predict.py#L25-L33
    """
    number_pos = np.sum(y_true, dtype=int)
    y_sort = np.flip(np.argsort(y_score))
    y_pred = np.zeros_like(y_true, dtype=int)
    y_pred[y_sort[np.arange(number_pos)]] = 1
    return y_pred
