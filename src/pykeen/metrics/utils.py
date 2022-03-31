# -*- coding: utf-8 -*-

"""Utilities for metrics."""

from dataclasses import dataclass
from typing import ClassVar, Collection, Iterable, Optional

import numpy as np
from docdata import get_docdata
from scipy import stats

from ..utils import camel_to_snake

__all__ = [
    "Metric",
    "ValueRange",
    "stable_product",
    "weighted_mean_expectation",
    "weighted_mean_variance",
    "weighted_harmonic_mean",
    "weighted_median",
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

    def approximate(self, epsilon: float) -> "ValueRange":
        """Create a slightly enlarged value range for approximate checks."""
        return ValueRange(
            lower=self.lower if self.lower is None else self.lower - epsilon,
            lower_inclusive=self.lower_inclusive,
            upper=self.upper if self.upper is None else self.upper + epsilon,
            upper_inclusive=self.upper_inclusive,
        )

    def notate(self) -> str:
        """Get the math notation for the range of this metric."""
        left = "(" if self.lower is None or not self.lower_inclusive else "["
        right = ")" if self.upper is None or not self.upper_inclusive else "]"
        return f"{left}{self._coerce(self.lower, low=True)}, {self._coerce(self.upper, low=False)}{right}"

    @staticmethod
    def _coerce(n: Optional[float], low: bool) -> str:
        if n is None:
            return "-inf" if low else "inf"  # ∞
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

    #: the value range
    value_range: ClassVar[ValueRange]

    #: synonyms for this metric
    synonyms: ClassVar[Collection[str]] = tuple()

    #: whether the metric supports weights
    supports_weights: ClassVar[bool] = False

    #: whether there is a closed-form solution of the expectation
    closed_expectation: ClassVar[bool] = False

    #: whether there is a closed-form solution of the variance
    closed_variance: ClassVar[bool] = False

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

    @classmethod
    def get_range(cls) -> str:
        """Get the math notation for the range of this metric."""
        docdata = get_docdata(cls) or {}
        left_bracket = "(" if cls.value_range.lower is None or not cls.value_range.lower_inclusive else "["
        left = docdata.get("tight_lower", cls.value_range._coerce(cls.value_range.lower, low=True))
        right_bracket = ")" if cls.value_range.upper is None or not cls.value_range.upper_inclusive else "]"
        right = docdata.get("tight_upper", cls.value_range._coerce(cls.value_range.upper, low=False))
        return f"{left_bracket}{left}, {right}{right_bracket}".replace("inf", "∞")

    def _extra_repr(self) -> Iterable[str]:
        return []

    def __repr__(self) -> str:  # noqa:D105
        return f"{self.__class__.__name__}({', '.join(self._extra_repr())})"


def weighted_mean_expectation(individual: np.ndarray, weights: Optional[np.ndarray]) -> float:
    r"""
    Calculate the expectation of a weighted sum of variables with given individual expected value.

    .. math::
        \mathbb{E}\left[\sum \limits_{i=1}^{n} w_i x_i\right]
            = \sum \limits_{i=1}^{n} w_i \mathbb{E}\left[x_i\right]

    where $w_i = \frac{1}{n}$, if no explicit weights are given. Moreover, the weights are normalized such that
    $\sum w_i = 1$.

    :param individual:
        the individual variables' expectations, $\mathbb{E}[x_i]$
    :param weights:
        the individual variables' weights

    :return:
        the variance of the weighted mean
    """
    return np.average(individual, weights=weights).item()


def weighted_mean_variance(individual: np.ndarray, weights: Optional[np.ndarray]) -> float:
    r"""
    Calculate the variance of a weighted mean of variables with given individual variances.

    .. math::
        \mathbb{V}\left[\sum \limits_{i=1}^{n} w_i x_i\right]
            = \sum \limits_{i=1}^{n} w_i^2 \mathbb{V}\left[x_i\right]

    where $w_i = \frac{1}{n}$, if no explicit weights are given. Moreover, the weights are normalized such that
    $\sum w_i = 1$.

    :param individual:
        the individual variables' variances, $\mathbb{V}[x_i]$
    :param weights:
        the individual variables' weights

    :return:
        the variance of the weighted mean
    """
    n = individual.size
    if weights is None:
        return individual.mean() / n
    weights = weights / weights.sum()
    return (individual * weights**2).sum().item()


def stable_product(a: np.ndarray, is_log: bool = False) -> np.ndarray:
    r"""
    Compute the product using the log-trick for increased numerical stability.

    .. math::

        \prod \limits_{i=1}^{n} a_i
            = \exp \log \prod \limits_{i=1}^{n} a_i
            = \exp \sum \limits_{i=1}^{n} \log a_i

    To support negative values, we additionally use

    .. math::

        a_i = \textit{sign}(a_i) * \textit{abs}(a_i)

    and

    .. math::

        \prod \limits_{i=1}^{n} a_i
            = \left(\prod \limits_{i=1}^{n} \textit{sign}(a_i)\right)
                \cdot \left(\prod \limits_{i=1}^{n} \textit{abs}(a_i)\right)

    where the first part is computed without the log-trick.

    :param a:
        the array
    :param is_log:
        whether the array already contains the logarithm of the elements

    :return:
        the product of elements
    """
    if is_log:
        sign = 1
    else:
        sign = np.prod(np.copysign(np.ones_like(a), a))
        a = np.log(np.abs(a))
    return sign * np.exp(np.sum(a))


def weighted_harmonic_mean(a: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate weighted harmonic mean.

    :param a:
        the array
    :param weights:
        the weight for individual array members

    :return:
        the weighted harmonic mean over the array

    .. seealso::
        https://en.wikipedia.org/wiki/Harmonic_mean#Weighted_harmonic_mean
    """
    if weights is None:
        return stats.hmean(a)

    # normalize weights
    weights = weights.astype(float)
    weights = weights / weights.sum()
    # calculate weighted harmonic mean
    return np.reciprocal(np.average(np.reciprocal(a.astype(float)), weights=weights))


def weighted_median(a: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate weighted median."""
    if weights is None:
        return np.median(a)

    # calculate cdf
    indices = np.argsort(a)
    s_ranks = a[indices]
    s_weights = weights[indices]
    cdf = np.cumsum(s_weights)
    cdf /= cdf[-1]
    # determine value at p=0.5
    idx = np.searchsorted(cdf, v=0.5)
    # special case for exactly 0.5
    if cdf[idx] == 0.5:
        return s_ranks[idx : idx + 2].mean()
    return s_ranks[idx]
