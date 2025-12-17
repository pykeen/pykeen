"""Utilities for metrics."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from docdata import get_docdata
from scipy import stats

from ..utils import ExtraReprMixin, camel_to_snake

__all__ = [
    "Metric",
    "ValueRange",
    "weighted_mean_expectation",
    "weighted_mean_variance",
    "weighted_harmonic_mean",
    "weighted_median",
    "compute_log_expected_power",
    "compute_median_survival_function",
]


@dataclass
class ValueRange:
    """A value range description."""

    #: the lower bound
    lower: float | None = None

    #: whether the lower bound is inclusive
    lower_inclusive: bool = False

    #: the upper bound
    upper: float | None = None

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
    def _coerce(n: float | None, low: bool) -> str:
        if n is None:
            return "-inf" if low else "inf"  # ∞
        if isinstance(n, int):
            return str(n)
        if n.is_integer():
            return str(int(n))
        return str(n)


class Metric(ExtraReprMixin):
    """A base class for metrics."""

    #: The name of the metric
    name: ClassVar[str]

    #: a link to further information
    link: ClassVar[str]

    #: whether the metric needs binarized scores
    binarize: ClassVar[bool | None] = None

    #: whether it is increasing, i.e., larger values are better
    increasing: ClassVar[bool]

    #: the value range
    value_range: ClassVar[ValueRange]

    #: synonyms for this metric
    synonyms: ClassVar[Collection[str]] = ()

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


def weighted_mean_expectation(individual: np.ndarray, weights: np.ndarray | None) -> float:
    r"""Calculate the expectation of a weighted mean of variables with given individual expected values.

    For random variables $x_1, \ldots, x_n$ with individual expectations
    $\mathbb{E}[x_i]$ and scalar weights $w_1, \ldots, w_n$, the expectation of the
    weighted mean is:

    .. math::

        \mathbb{E}\left[\frac{\sum \limits_{i=1}^{n} w_i x_i}{\sum \limits_{j=1}^{n} w_j}\right]
            = \frac{\sum \limits_{i=1}^{n} w_i \mathbb{E}\left[x_i\right]}{\sum \limits_{j=1}^{n} w_j}

    When $w_i = \frac{1}{n}$ (uniform weights, used if no explicit weights are given),
    the weights are normalized such that $\sum w_i = 1$.

    .. note::

        Unlike variance, the expected value formula is identical for both scaling factor
        and repeat count interpretations of weights.

    :param individual: the individual variables' expectations, $\mathbb{E}[x_i]$
    :param weights: the individual variables' scalar weights

    :returns: the expectation of the weighted mean
    """
    return np.average(individual, weights=weights).item()


def weighted_mean_variance(individual: np.ndarray, weights: np.ndarray | None) -> float:
    r"""Calculate the variance of a weighted mean of variables with given individual variances.

    For independent random variables $x_1, \ldots, x_n$ with individual variances
    $\mathbb{V}[x_i]$ and arbitrary scalar weights $w_1, \ldots, w_n$, the variance of
    the weighted mean is:

    .. math::

        \mathbb{V}\left[\frac{\sum \limits_{i=1}^{n} w_i x_i}{\sum \limits_{j=1}^{n} w_j}\right]
            = \frac{\sum \limits_{i=1}^{n} w_i^2 \mathbb{V}\left[x_i\right]}{\left(\sum \limits_{j=1}^{n} w_j\right)^2}

    The $w_i^2$ term arises from the variance scaling property: $\mathbb{V}[c \cdot X] =
    c^2 \cdot \mathbb{V}[X]$.

    When $w_i = \frac{1}{n}$ (uniform weights, used if no explicit weights are given),
    the weights are normalized such that $\sum w_i = 1$.

    .. note::

        This implements **scaling factor semantics**: each variable is sampled once and
        scaled by its weight. This differs from **repeat count semantics** where weights
        would represent the number of independent samples, which would yield a linear
        (not quadratic) dependence on weights.

    :param individual: the individual variables' variances, $\mathbb{V}[x_i]$
    :param weights: the individual variables' scalar weights (not repeat counts)

    :returns: the variance of the weighted mean
    """
    n = individual.size
    if weights is None:
        return individual.mean() / n
    total_weight = weights.sum()
    return (individual * weights**2).sum().item() / (total_weight**2)


def weighted_harmonic_mean(a: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Calculate weighted harmonic mean.

    :param a: the array
    :param weights: the weight for individual array members

    :returns: the weighted harmonic mean over the array

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


def weighted_median(a: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Calculate weighted median."""
    if weights is None:
        return np.median(a)

    # calculate cdf
    indices = np.argsort(a)
    s_ranks = a[indices]
    s_weights = weights[indices]
    cdf = np.cumsum(s_weights)
    # to avoid loss of precision, we do not normalize, but keep the original data type
    # but, we have to adjust the 0.5 search value accordingly
    # cdf = cdf / cdf[-1]
    # idx = np.searchsorted(cdf, v=0.5)
    v = 0.5 * cdf[-1]
    idx = np.searchsorted(cdf, v=v)
    # special case for exactly 0.5
    if cdf[idx] == v:
        return s_ranks[idx : idx + 2].mean()
    return s_ranks[idx]


def compute_log_expected_power(k_values: np.ndarray, powers: np.ndarray, memory_limit_elements: int = 10**7) -> float:
    """
    Compute $sum( ln( E[X_i^p_i] ) )$.

    Does so efficiently by using sorted batching and vectorization.

    :param k_values: shape: (n,)
        Upper bounds.
    :param powers: shape: (n,)
        Exponents.
    :param memory_limit_elements:
        Max number of float elements in the temporary matrix buffer.
        10^7 elements ~ 80MB RAM.

    :return:
        The scalar log-value.
    """
    # 1. Sort inputs by k.
    # This minimizes the 'masking waste' when we batch variables together.
    # Small k's are processed with small k's; large with large.
    sort_idx = np.argsort(k_values)
    k_sorted = np.asarray(k_values)[sort_idx]
    p_sorted = np.asarray(powers)[sort_idx]

    n = len(k_sorted)
    total_log_moment = 0.0

    start_idx = 0
    while start_idx < n:
        # 2. Determine Batch Size dynamically based on Memory Limit.
        # We want to find 'end_idx' such that:
        # (rows in batch) * (max_k in batch) <= memory_limit
        # Since k is sorted ascending, max_k_in_batch is simply k_sorted[end_idx-1].

        end_idx = start_idx + 1

        # We peek ahead to see how many rows we can fit.
        # This is a heuristic: we check if adding the next block of rows
        # would explode the matrix size required.
        while end_idx < n:
            current_max_k = k_sorted[end_idx]  # This would be the new width
            current_rows = end_idx - start_idx + 1

            # Check budget
            if current_rows * current_max_k > memory_limit_elements:
                break
            end_idx += 1

        # 3. Process the Batch
        k_batch = k_sorted[start_idx:end_idx]
        p_batch = p_sorted[start_idx:end_idx]

        # Max k in this specific batch (determines matrix columns)
        max_k_batch = int(k_batch[-1])

        # TODO: check if we are 0- or 1-based
        # Create base grid [1, 2, ..., max_k_batch]
        # Shape: (1, max_k)
        j_grid = np.arange(1, max_k_batch + 1, dtype=np.float64).reshape(1, -1)

        # Reshape powers for broadcasting
        # Shape: (batch_rows, 1)
        p_col = p_batch.reshape(-1, 1)

        # 4. Compute Powers (Broadcasting)
        # Matrix Result: (batch_rows, max_k)
        # value[i, j] = (j+1) ^ p_i
        values = j_grid**p_col

        # 5. Masking
        # Since k is sorted, we have a "lower triangular" style validity mask
        # (roughly), but strictly we just need to zero out j > k_i.
        # Shape: (batch_rows, 1) compared to (1, max_k)
        mask = j_grid <= k_batch.reshape(-1, 1)

        # Apply mask (zero out values strictly greater than k_i)
        # We use 'where' or multiplication. Multiplication is faster for floats usually.
        values *= mask

        # 6. Summation
        # Sum along columns to get sum(j^p) for each variable
        row_sums = np.sum(values, axis=1)

        # Compute log terms: log(sum) - log(k)
        # We use np.log with a safety for the 0 sums (though sums of j^p >= 1 are never 0)
        batch_log_moments = np.log(row_sums) - np.log(k_batch)

        total_log_moment += np.sum(batch_log_moments)

        # Move to next batch
        start_idx = end_idx

    return total_log_moment


def compute_median_survival_function(num_candidates: np.ndarray) -> np.ndarray:
    """Compute $P(Median > x)$ for x in range $[0, max(k)]$.

    This function uses dynamic programming to calculate the cumulative distribution
    of the count of variables <= x, thereby deriving the median's distribution.

    Memory complexity: O(K * n), where K is the maximum value of k.

    :param num_candidates: shape: (n,)
        The number of candidates.

    :return: shape: (K,)
        The survival function. $K$ denotes the maximum number of candidates.
    """
    ks = np.array(num_candidates, dtype=int)
    n = len(ks)
    k_max = ks.max()

    # We target the index n // 2.
    # For n=3 (odd), index 1 (2nd smallest).
    # For n=4 (even), index 2 (3rd smallest, i.e., the 'upper' median).
    target_threshold = n // 2

    # Grid of values x = 0, 1, ..., k_max
    # We compute probabilities up to k_max.
    x_grid = np.arange(k_max + 1)

    # Matrix of individual probabilities: P(X_i <= x)
    # Shape: (k_max + 1, n)
    # P(X_i <= x) = min(1, x / k_i)
    p_matrix = np.minimum(1.0, x_grid[:, None] / ks[None, :])

    # DP State: dp[v, c] = Probability that exactly 'c' variables are <= v
    # Initialize: 0 variables <= v has probability 1 initially
    dp = np.zeros((len(x_grid), n + 1))
    dp[:, 0] = 1.0

    # Vectorized Poisson-Binomial recurrence
    for i in range(n):
        p = p_matrix[:, i : i + 1]  # Column vector for broadcasting

        # New DP state based on convolution with Bernoulli(p)
        # dp[c] = dp[c]*(1-p) + dp[c-1]*p
        term_fail = dp * (1 - p)

        term_success = np.zeros_like(dp)
        term_success[:, 1:] = dp[:, :-1] * p

        dp = term_fail + term_success

    # The median is <= x if the count of variables (<= x) is > target_threshold.
    # CDF(x) = P(Median <= x) = Sum_{c=target+1}^{n} P(Count == c)
    cdf_median = dp[:, target_threshold + 1 :].sum(axis=1)

    # Survival Function: P(Median > x) = 1 - CDF(x)
    return 1.0 - cdf_median
