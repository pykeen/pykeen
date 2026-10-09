"""Utilities for metrics."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from docdata import get_docdata
from scipy import fft, special, stats

from ..utils import ExtraReprMixin, camel_to_snake

__all__ = [
    "Metric",
    "ValueRange",
    "compute_log_expected_power",
    "compute_median_mean",
    "compute_median_moments",
    "compute_median_survival_function",
    "compute_order_statistic_survival_function",
    "weighted_harmonic_mean",
    "weighted_mean_expectation",
    "weighted_mean_variance",
    "weighted_median",
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
        if cls.__doc__ is None:
            raise ValueError(f"{cls.__name__} has neither a docdata description nor a docstring.")
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
    return (individual * (weights / weights.sum()) ** 2).sum().item()


def stable_product(a: np.ndarray, is_log: bool = False) -> np.ndarray:
    r"""Compute the product using the log-trick for increased numerical stability.

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

    :param a: the array
    :param is_log: whether the array already contains the logarithm of the elements

    :returns: the product of elements
    """
    if is_log:
        sign = 1
    else:
        sign = np.prod(np.copysign(np.ones_like(a), a))
        a = np.log(np.abs(a))
    return sign * np.exp(np.sum(a))


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


def weighted_median(a: np.ndarray, weights: np.ndarray | None = None) -> np.floating:
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


def compute_log_expected_power(k_values: np.ndarray, powers: np.ndarray, memory_limit_elements: int = 10**7) -> float:
    r"""
    Compute $\sum_i \ln \mathbb{E}[X_i^{p_i}]$ for independent $X_i$ uniformly distributed on $\{1, \ldots, k_i\}$.

    Each summand is given by

    .. math::

        \ln \mathbb{E}[X_i^{p_i}] = \ln \sum \limits_{j=1}^{k_i} \exp(p_i \cdot \ln j) - \ln k_i

    For each unique power $p$, the log-sums for all $k \leq \max_i k_i$ are obtained at once by a single cumulative sum,
    and then looked up for each $k_i$. Hence, for $u$ unique powers, the cost is $\mathcal{O}(u \cdot \max_i k_i + n)$
    rather than $\mathcal{O}(n \cdot \max_i k_i)$. In particular, $u = 1$ for the unweighted geometric mean rank.

    :param k_values: shape: (n,)
        Upper bounds.
    :param powers: shape: (n,)
        Exponents.
    :param memory_limit_elements:
        Max number of float elements in the temporary buffer of log-cumulative sums, which has one row per unique power.
        10^7 elements ~ 80MB RAM.

    :return:
        The scalar log-value.
    """
    k_values = np.asarray(k_values).astype(np.int64, copy=False)
    if (k_values < 1).any():
        raise ValueError(f"All upper bounds must be at least 1, but the minimum is {k_values.min()}.")
    unique_powers, inverse = np.unique(np.asarray(powers, dtype=np.float64), return_inverse=True)
    inverse = inverse.reshape(-1)
    num_unique = len(unique_powers)

    # the largest k per unique power, which determines the length of its log-cumulative sum
    max_k = np.zeros(num_unique, dtype=np.int64)
    np.maximum.at(max_k, inverse, k_values)

    # process unique powers sorted by their largest k, to reduce padding when batching them
    order = np.argsort(max_k, kind="stable")
    max_k_sorted = max_k[order]
    # position of each unique power in the processing order
    position = np.empty(num_unique, dtype=np.int64)
    position[order] = np.arange(num_unique)
    task_position = position[inverse]

    # log sum_{j=1}^{k_i} j^{p_i} for each task
    log_sums = np.empty(len(k_values), dtype=np.float64)
    start = 0
    while start < num_unique:
        # choose the batch such that (number of rows) * (largest k) stays within the memory limit (but >= 1 row)
        # note: since max_k_sorted is ascending, the required size increases with the number of rows
        sizes = np.arange(1, num_unique - start + 1) * max_k_sorted[start:]
        stop = start + max(1, int(np.searchsorted(sizes, memory_limit_elements, side="right")))
        batch_powers = unique_powers[order[start:stop], None]
        log_j = np.log(np.arange(1, max_k_sorted[stop - 1] + 1, dtype=np.float64))
        # shape: (stop - start, max_k); entry [b, k - 1] = exp(-shift_b) * sum_{j=1}^{k} j^{p_b}
        # note: we shift by the largest exponent per row, p * log(max_k) for p > 0, and 0 otherwise, to avoid overflow;
        #       this is faster than a log-cumsum-exp, which matters when there are many unique powers.
        shift = np.maximum(batch_powers * log_j[-1], 0.0)
        table = np.cumsum(np.exp(batch_powers * log_j[None, :] - shift), axis=1)
        # look up the log-sums for all tasks whose power is in this batch
        task_idx = np.flatnonzero((task_position >= start) & (task_position < stop))
        rows = task_position[task_idx] - start
        cols = k_values[task_idx] - 1
        with np.errstate(divide="ignore"):  # log(0) for underflowed entries, which are replaced below
            log_sums[task_idx] = np.log(table[rows, cols]) + shift[rows, 0]
        # for large powers, small j underflow after shifting, i.e., table[:, 0] == 0; as the cumulative sums are
        # monotone, only these rows are affected. They are rare, so we recompute them exactly in log-space.
        underflowed = table[:, 0] == 0.0
        if underflowed.any():
            log_table = np.logaddexp.accumulate(batch_powers[underflowed] * log_j[None, :], axis=1)
            row_to_log_row = np.cumsum(underflowed) - 1
            selected = underflowed[rows]
            log_sums[task_idx[selected]] = log_table[row_to_log_row[rows[selected]], cols[selected]]
        start = stop

    return float(np.sum(log_sums - np.log(k_values)))


#: maximum number of array elements to hold per chunk when evaluating count distributions
_CHUNK_ELEMENTS = 2_000_000

#: maximum amount of work (in complex multiplications) for the exact pairwise term of the even-n median variance
#: with multiple distinct numbers of candidates
_MAX_PAIRWISE_WORK = 5e7


def _group_candidates(num_candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group ranking tasks by their number of candidates.

    :param num_candidates: shape: (n,)
        The number of candidates for each ranking task.

    :return: shape: (g,), (g,)
        The unique numbers of candidates, and the number of tasks for each of them.
    """
    ks, counts = np.unique(np.asarray(num_candidates, dtype=int), return_counts=True)
    return ks, counts


def _count_cdf(ks: np.ndarray, counts: np.ndarray, x_grid: np.ndarray, c: int) -> np.ndarray:
    r"""Compute $P(C(x) \leq c)$, where $C(x) = |\{i : r_i \leq x\}|$ for $r_i \sim \mathcal{U}(1, k_i)$.

    The count is a sum of independent binomials, one for each group of tasks with identical $k$, with success
    probability $\min(1, x / k)$.

    For a single group, this is a vectorized evaluation of the binomial CDF. For multiple groups, we convolve the
    probability mass functions of the groups using FFTs along the count axis, vectorized over chunks of $x$.

    :param ks: shape: (g,)
        The unique numbers of candidates.
    :param counts: shape: (g,)
        The number of tasks for each unique number of candidates.
    :param x_grid: shape: (m,)
        The values $x$.
    :param c:
        The (inclusive) upper limit for the count.

    :return: shape: (m,)
        The probabilities.
    """
    n = int(counts.sum())
    if c >= n:
        return np.ones(len(x_grid))
    if len(ks) == 1:
        return stats.binom.cdf(c, n, np.minimum(1.0, x_grid / ks[0]))
    size = fft.next_fast_len(n + 1)
    chunk = max(1, _CHUNK_ELEMENTS // size)
    result = np.empty(len(x_grid))
    for start in range(0, len(x_grid), chunk):
        x = x_grid[start : start + chunk]
        spectrum = None
        for k, n_g in zip(ks, counts, strict=True):
            p = np.minimum(1.0, x / k)
            pmf = stats.binom.pmf(np.arange(n_g + 1)[None, :], n_g, p[:, None])
            spec = fft.rfft(pmf, n=size, axis=-1)
            spectrum = spec if spectrum is None else spectrum * spec
        pmf = fft.irfft(spectrum, n=size, axis=-1)[:, : c + 1]
        result[start : start + chunk] = np.clip(pmf, 0.0, None).sum(axis=-1)
    return np.clip(result, 0.0, 1.0)


def compute_order_statistic_survival_function(num_candidates: np.ndarray, index: int) -> np.ndarray:
    r"""Compute $P(X_{(j)} > x)$ for the $j$-th smallest of independent $r_i \sim \mathcal{U}(1, k_i)$.

    Let $C(x) = |\{i : r_i \leq x\}|$. Then, $X_{(j)} > x$ if and only if $C(x) \leq j - 1$. The distribution of
    $C(x)$ is a convolution of binomial distributions, one for each group of tasks with identical number of
    candidates.

    Time complexity: $O(K)$ for a single unique number of candidates; for $g$ unique values, we need
    $O(K \cdot g \cdot n \log n)$. Memory is bounded by a constant, since the grid is processed in chunks.

    :param num_candidates: shape: (n,)
        The number of candidates.
    :param index:
        The 1-based index $j \in \{1, \ldots, n\}$ of the order statistic.

    :return: shape: (K + 1,)
        The survival function for $x = 0, \ldots, K$, where $K$ denotes the maximum number of candidates.

    :raises ValueError:
        If the index is out of range.
    """
    ks, counts = _group_candidates(num_candidates)
    if not 1 <= index <= counts.sum():
        raise ValueError(f"index must be in [1, {counts.sum()}], but is {index}")
    x_grid = np.arange(ks.max() + 1)
    return _count_cdf(ks, counts, x_grid, c=index - 1)


def compute_median_survival_function(num_candidates: np.ndarray) -> np.ndarray:
    r"""Compute $P(M > x)$ for $x \in \{0, \ldots, K\}$, where $M$ is the (upper) median of the ranks.

    For an odd number $n$ of ranks, $M$ is the median, i.e., the order statistic $X_{(\frac{n + 1}{2})}$. For an even
    number of ranks, the median as computed by :func:`numpy.median` is the average of the two middle order statistics
    and is not integer-valued. This function then returns the survival function of the *upper* median
    $X_{(\frac{n}{2} + 1)}$ instead. Use :func:`compute_order_statistic_survival_function` for the lower one, and
    :func:`compute_median_mean` / :func:`compute_median_moments` for the mean and variance of the actual median.

    Time complexity: $O(K)$ for a single unique number of candidates, otherwise $O(K \cdot g \cdot n \log n)$, where
    $g$ is the number of unique numbers of candidates. Memory is bounded by a constant.

    :param num_candidates: shape: (n,)
        The number of candidates.

    :return: shape: (K + 1,)
        The survival function. $K$ denotes the maximum number of candidates.
    """
    n = len(num_candidates)
    return compute_order_statistic_survival_function(num_candidates, index=n // 2 + 1)


def _pairwise_term(ks: np.ndarray, counts: np.ndarray, m: int) -> float:
    r"""Compute $\sum_{0 \leq a < b} P(C(a) = m, C(b) = m)$ for $n = 2m$.

    The event means that exactly $m$ ranks are $\leq a$ and none is in $(a, b]$, i.e.,

    .. math::

        P(C(a) = m, C(b) = m) = [z^m] \prod_g \left(p_g(a) z + 1 - p_g(b)\right)^{n_g}

    For a single group, this is $\binom{n}{m} p(a)^m (1 - p(b))^{n - m}$, which is separable in $a$ and $b$ and can
    thus be summed in $O(K)$ using a cumulative log-sum-exp. For multiple groups, we extract the coefficient via a
    DFT over evaluations at roots of unity, which needs $O(K^2 \cdot g \cdot n)$ operations.

    :param ks: shape: (g,)
        The unique numbers of candidates.
    :param counts: shape: (g,)
        The number of tasks for each unique number of candidates.
    :param m:
        Half of the number of tasks.

    :return:
        The sum.

    :raises NotImplementedError:
        If there are multiple groups and the computation exceeds the work limit.
    """
    n = int(counts.sum())
    k_max = int(ks.max())
    if len(ks) == 1:
        x = np.arange(k_max + 1)
        p = np.minimum(1.0, x / ks[0])
        with np.errstate(divide="ignore"):
            log_p = np.log(p)
            log_q = np.log1p(-p)
        # sum_{a < b} u_a v_b = sum_b v_b * (sum_{a < b} u_a), in log-space
        log_cum = np.logaddexp.accumulate(m * log_p)
        log_binom = special.gammaln(n + 1) - 2 * special.gammaln(m + 1)
        # terms for b = 1, ..., K
        return float(np.exp(log_binom + log_cum[:-1] + m * log_q[1:]).sum())

    size = n + 1
    work = 0.5 * k_max**2 * len(ks) * size
    if work > _MAX_PAIRWISE_WORK:
        raise NotImplementedError(
            f"Exact computation requires ~{work:.2g} operations, which exceeds the limit of {_MAX_PAIRWISE_WORK:.2g}."
        )
    omega = np.exp(2j * np.pi * np.arange(size) / size)
    phase = omega ** (-m)
    # shape: (g, K + 1)
    x = np.arange(k_max + 1)
    p = np.minimum(1.0, x[None, :] / ks[:, None])
    q = 1.0 - p
    chunk = max(1, _CHUNK_ELEMENTS // (size * k_max))
    total = 0.0
    for start in range(1, k_max, chunk):
        a = np.arange(start, min(start + chunk, k_max))
        b = np.arange(start + 1, k_max + 1)
        acc = np.ones((len(a), len(b), size), dtype=complex)
        for g, n_g in enumerate(counts):
            acc *= (p[g, a][:, None, None] * omega[None, None, :] + q[g, b][None, :, None]) ** n_g
        values = (acc * phase).sum(axis=-1).real / size
        mask = b[None, :] > a[:, None]
        total += float(np.clip(values, 0.0, None)[mask].sum())
    return total


def compute_median_mean(num_candidates: np.ndarray) -> float:
    r"""Compute the exact expected value of the median of independent $r_i \sim \mathcal{U}(1, k_i)$.

    For odd $n$ this is $\sum_{x \geq 0} P(X_{(\frac{n+1}{2})} > x)$, for even $n$ the average of the corresponding
    sums for the two middle order statistics. It needs $O(K)$ for a single unique number of candidates, and
    $O(K \cdot g \cdot n \log n)$ for $g$ unique values.

    :param num_candidates: shape: (n,)
        The number of candidates.

    :return:
        The expected value of the median.
    """
    n = len(num_candidates)
    indices = [n // 2 + 1] if n % 2 == 1 else [n // 2, n // 2 + 1]
    return float(
        np.mean([compute_order_statistic_survival_function(num_candidates, index=i)[:-1].sum() for i in indices])
    )


def compute_median_moments(num_candidates: np.ndarray) -> tuple[float, float]:
    r"""Compute the exact mean and variance of the median of independent $r_i \sim \mathcal{U}(1, k_i)$.

    The median follows :func:`numpy.median`: for odd $n$, it is the order statistic $X_{(\frac{n+1}{2})}$, and for
    even $n$ the average $M = \frac{A + B}{2}$ of $A = X_{(m)}$ and $B = X_{(m + 1)}$ with $m = \frac{n}{2}$. We use
    $\mathbb{E}[X] = \sum_{x \geq 0} P(X > x)$ and $\mathbb{E}[X^2] = \sum_{x \geq 0} (2x + 1) P(X > x)$ for
    non-negative integer random variables. For even $n$ we additionally need

    .. math::

        \mathbb{E}[AB] = \sum_{a \geq 0} \sum_{b \geq 0} P(A > a, B > b)

    Since $A \leq B$, the summand is $P(A > a)$ for $b \leq a$. For $b > a$, with $C(x) = |\{i : r_i \leq x\}|$,

    .. math::

        P(A > a, B > b) = P(C(b) \leq m) - P(C(a) = m, C(b) = m)

    The first term only depends on $b$, the second one is summed by :func:`_pairwise_term`.

    Complexity: for a single unique number of candidates, $O(K)$. For $g$ unique values, odd $n$ needs
    $O(K \cdot g \cdot n \log n)$. For even $n$, the exact variance additionally needs $O(K^2 \cdot g \cdot n)$,
    which is only attempted if it is below a fixed work limit.

    :param num_candidates: shape: (n,)
        The number of candidates.

    :return:
        The mean and the variance of the median. The variance is clamped to be non-negative.

    :raises NotImplementedError:
        If $n$ is even, there are multiple unique numbers of candidates, and the exact variance is too expensive to
        compute. The mean is always available via :func:`compute_median_mean`.
    """
    ks, counts = _group_candidates(num_candidates)
    n = int(counts.sum())
    if n % 2 == 1:
        sf = compute_median_survival_function(num_candidates)[:-1]
        x = np.arange(len(sf))
        mean = float(sf.sum())
        return mean, max(0.0, float(((2 * x + 1) * sf).sum()) - mean**2)
    m = n // 2
    sf_a = compute_order_statistic_survival_function(num_candidates, index=m)[:-1]
    sf_b = compute_order_statistic_survival_function(num_candidates, index=m + 1)[:-1]
    x = np.arange(len(sf_a))
    mean = 0.5 * float(sf_a.sum() + sf_b.sum())
    e_aa = float(((2 * x + 1) * sf_a).sum())
    e_bb = float(((2 * x + 1) * sf_b).sum())
    e_ab = float(((x + 1) * sf_a).sum() + (x * sf_b).sum()) - _pairwise_term(ks, counts, m)
    return mean, max(0.0, 0.25 * (e_aa + e_bb + 2 * e_ab) - mean**2)
