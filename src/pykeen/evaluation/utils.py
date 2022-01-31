# -*- coding: utf-8 -*-

"""Utilities for evaluation."""

from typing import Callable, MutableMapping, NamedTuple, Optional, Union

import numpy as np
from rexmex.utils import binarize as binarize_func

__all__ = [
    "MetricAnnotation",
    "MetricAnnotator",
]


class MetricAnnotation(NamedTuple):
    """Metadata about a classifier function."""

    func: Callable[[np.array, np.array], float]
    type: str
    name: str
    lower: float
    upper: float
    higher_is_better: bool
    description: str
    link: str
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def interval(self) -> str:
        """Get the math notation for the range of this metric."""
        left = "[" if self.lower_inclusive else "("
        right = "]" if self.upper_inclusive else ")"
        lower: Union[int, float]
        upper: Union[int, float]
        try:
            lower = int(self.lower)
        except OverflowError:
            lower = self.lower
            left = "("
        try:
            upper = int(self.upper)
        except OverflowError:
            upper = self.upper
            right = ")"
        return f"{left}{lower}, {upper}{right}"


class MetricAnnotator:
    """A class for annotating metric functions."""

    type: str
    metrics: MutableMapping[str, MetricAnnotation]

    def __init__(self, label: str):
        self.type = label
        self.metrics = {}

    def higher(self, func, **kwargs):
        """Annotate a function where higher values are better."""
        kwargs["higher_is_better"] = True
        return self.add(func, **kwargs)

    def lower(self, func, **kwargs):
        """Annotate a function where lower values are better."""
        kwargs["higher_is_better"] = False
        return self.add(func, **kwargs)

    def add(
        self,
        func,
        *,
        higher_is_better: bool,
        description: str,
        link: str,
        name: Optional[str] = None,
        lower: float = 0.0,
        lower_inclusive: bool = True,
        upper: float = 1.0,
        upper_inclusive: bool = True,
        binarize: bool = False,
    ):
        """Annotate a function."""
        self.metrics[func] = MetricAnnotation(
            func=binarize_func(func) if binarize else func,
            type=self.type,
            name=name or func.__name__.replace("_", " ").title(),
            lower=lower,
            lower_inclusive=lower_inclusive,
            upper=upper,
            upper_inclusive=upper_inclusive,
            higher_is_better=higher_is_better,
            description=description,
            link=link,
        )
