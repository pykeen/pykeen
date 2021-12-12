# -*- coding: utf-8 -*-

"""Utilities for evaluation."""

from typing import Callable, MutableMapping, NamedTuple, Optional

import numpy as np

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
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    description: Optional[str] = None

    def range_str(self) -> str:
        """Get the math notation for the range of this metric."""
        left = "[" if self.lower_inclusive else "("
        right = "]" if self.upper_inclusive else ")"
        try:
            lower = int(self.lower)
        except OverflowError:
            lower = self.lower
        try:
            upper = int(self.upper)
        except OverflowError:
            upper = self.upper
        return f"{left}{lower}, {upper}{right}"

    def get_doc(self) -> str:
        """Get the documentation string for this metric."""
        rv = ""
        if self.description:
            rv += self.description + " "
        rv += f" On {self.range_str()}, "
        return rv


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
        name: Optional[str] = None,
        description: Optional[str] = None,
        lower: float = 0.0,
        lower_inclusive: bool = True,
        upper: float = 1.0,
        upper_inclusive: bool = True,
    ):
        """Annotate a function."""
        self.metrics[func] = MetricAnnotation(
            func=func,
            type=self.type,
            name=name or func.__name__.replace("_", " ").title(),
            lower=lower,
            lower_inclusive=lower_inclusive,
            upper=upper,
            upper_inclusive=upper_inclusive,
            higher_is_better=higher_is_better,
            description=description,
        )
