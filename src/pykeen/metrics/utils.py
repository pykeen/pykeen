# -*- coding: utf-8 -*-

"""Utilities for metrics."""

from dataclasses import dataclass
from typing import Callable, ClassVar, Collection, Iterable, MutableMapping, Optional

import numpy as np
from docdata import get_docdata

from ..utils import camel_to_snake

__all__ = [
    "Metric",
    "MetricAnnotation",
    "MetricAnnotator",
    "construct_indicator",
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


class Metric:
    """A base class for metrics."""

    #: The name of the metric
    name: ClassVar[str]

    #: a link to further information
    link: ClassVar[str]

    #: whether the metric needs binarized scores
    binarize: ClassVar[Optional[bool]] = None

    #: whether it is increasing, i.e., larger values are better
    increasing: ClassVar[bool]

    #: the value range (as string)
    value_range: ClassVar[Optional[ValueRange]] = None

    #: synonyms for this metric
    synonyms: ClassVar[Collection[str]] = tuple()

    @classmethod
    def get_description(cls) -> str:
        """Get the description."""
        docdata = get_docdata(cls)
        if docdata is not None and "description" in docdata:
            return docdata["description"]
        assert cls.__doc__ is not None
        return cls.__doc__.splitlines()[0]

    @classmethod
    def get_link(cls) -> str:
        """Get the link from the docdata."""
        docdata = get_docdata(cls)
        if docdata is None:
            raise TypeError
        return docdata["link"]

    @property
    def key(self) -> str:
        """Return the key for use in metric result dictionaries."""
        return camel_to_snake(self.__class__.__name__)

    def _extra_repr(self) -> Iterable[str]:
        return []

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(self._extra_repr())})"


@dataclass
class MetricAnnotation:
    """Metadata about a classifier function."""

    name: str
    increasing: bool
    value_range: ValueRange
    description: str
    link: str
    func: Callable[[np.array, np.array], float]
    binarize: bool

    def score(self, y_true, y_score) -> float:
        """Run the scoring function."""
        if self.func is None:
            raise ValueError
        return self.func(y_true, construct_indicator(y_score=y_score, y_true=y_true) if self.binarize else y_score)


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

    .. note ::
        Since the method uses the number of true labels to determine a threshold, the
        results will typically be overly optimistic estimates of the generalization performance.

    .. todo ::
        Add a method which estimates a threshold based on a validation set, and applies this
        threshold for binarization on the test set.

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
