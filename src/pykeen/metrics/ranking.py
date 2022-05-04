# -*- coding: utf-8 -*-

"""
Ranking metrics.

This module comprises various rank-based metrics, which get an array of individual ranks as input, as summarize them
into a single-figure metric measuring different aspects of ranking performance.

We can generally distinguish:

Base Metrics
------------
These metrics directly operate on the ranks:

The following metrics measures summarize the central tendency of ranks

- :class:`pykeen.metrics.ranking.ArithmeticMeanRank`
- :class:`pykeen.metrics.ranking.GeometricMeanRank`
- :class:`pykeen.metrics.ranking.HarmonicMeanRank`
- :class:`pykeen.metrics.ranking.MedianRank`

The Hits at K metric is closely related to information retrieval and measures the fraction of times when the correct
result is in the top-$k$ ranked entries, i.e., the rank is at most $k$

- :class:`pykeen.metrics.ranking.HitsAtK`

The next metrics summarize the dispersion of ranks

- :class:`pykeen.metrics.ranking.MedianAbsoluteDeviation`
- :class:`pykeen.metrics.ranking.Variance`
- :class:`pykeen.metrics.ranking.StandardDeviation`

and finally there is a simple metric to store the number of ranks which where aggregated

- :class:`pykeen.metrics.ranking.Count`

Inverse Metrics
---------------
The inverse metrics are reciprocals of the central tendency measures. They offer the advantage of having a fixed value
range of $(0, 1]$, with a known optimal value of $1$:

- :class:`pykeen.metrics.ranking.InverseArithmeticMeanRank`
- :class:`pykeen.metrics.ranking.InverseGeometricMeanRank`
- :class:`pykeen.metrics.ranking.InverseHarmonicMeanRank`
- :class:`pykeen.metrics.ranking.InverseMedianRank`

Adjusted Metrics
----------------
Adjusted metrics build upon base metrics, but adjust them for chance, cf. [berrendorf2020]_ and [hoyt2022]_. All
adjusted metrics derive from :class:`pykeen.metrics.ranking.DerivedRankBasedMetric` and, for a given evaluation set,
are affine transformations of the base metric with dataset-dependent, but fixed transformation constants. Thus, they
can also be computed when the model predictions are not available anymore, but the evaluation set is known.

Expectation-Normalized Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These metrics divide the metric by its expected value under random ordering. Thus, their expected value is always 1
irrespective of the evaluation set. They derive from :class:`pykeen.metrics.ranking.ExpectationNormalizedMetric`, and
there is currently only a single implementation:

- :class:`pykeen.metrics.ranking.AdjustedArithmeticMeanRank`

Re-indexed Metrics
~~~~~~~~~~~~~~~~~~
Re-indexed metrics subtract the expected value, and then normalize the optimal value to be 1. Thus, their expected value
under random ordering is 0, their optimal value is 1, and larger values indicate better results. The classes derive from
:class:`pykeen.metrics.ranking.ReindexedMetric`, and the following implementations are available:

- :class:`pykeen.metrics.ranking.AdjustedHitsAtK`
- :class:`pykeen.metrics.ranking.AdjustedArithmeticMeanRankIndex`
- :class:`pykeen.metrics.ranking.AdjustedGeometricMeanRankIndex`
- :class:`pykeen.metrics.ranking.AdjustedInverseHarmonicMeanRank`

z-Adjusted Metrics
~~~~~~~~~~~~~~~~~~
The final type of adjusted metrics uses the expected value as well as the variance of the metric under random ordering
to normalize the metrics similar to `z-score normalization <https://en.wikipedia.org/wiki/Standard_score>`_.
The z-score normalized metrics have an expected value of 0, and a variance of 1, and positive values indicate better
results. While their value range is unbound, it can be interpreted through the lens of the inverse cumulative
density function of the standard Gaussian distribution to retrieve a *p*-value. The classes derive from
:class:`pykeen.metrics.ranking.ZMetric`, and the following implementations are available:

- :class:`pykeen.metrics.ranking.ZArithmeticMeanRank`
- :class:`pykeen.metrics.ranking.ZGeometricMeanRank`
- :class:`pykeen.metrics.ranking.ZHitsAtK`
- :class:`pykeen.metrics.ranking.ZInverseHarmonicMeanRank`
"""
import math
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Collection, Iterable, NamedTuple, Optional, Tuple, Type, Union

import numpy as np
from class_resolver import ClassResolver, HintOrType
from docdata import parse_docdata
from scipy import stats

from .utils import (
    Metric,
    ValueRange,
    stable_product,
    weighted_harmonic_mean,
    weighted_mean_expectation,
    weighted_mean_variance,
    weighted_median,
)
from ..typing import RANK_REALISTIC, RANK_TYPES, RankType
from ..utils import logcumsumexp

__all__ = [
    "rank_based_metric_resolver",
    # Base classes
    "RankBasedMetric",
    "DerivedRankBasedMetric",
    "ExpectationNormalizedMetric",
    "ReindexedMetric",
    "ZMetric",
    # Concrete classes
    "ArithmeticMeanRank",
    "AdjustedArithmeticMeanRank",
    "AdjustedArithmeticMeanRankIndex",
    "ZArithmeticMeanRank",
    "InverseArithmeticMeanRank",
    #
    "GeometricMeanRank",
    "AdjustedGeometricMeanRankIndex",
    "ZGeometricMeanRank",
    "InverseGeometricMeanRank",
    #
    "HarmonicMeanRank",
    "InverseHarmonicMeanRank",
    "AdjustedInverseHarmonicMeanRank",
    "ZInverseHarmonicMeanRank",
    #
    "MedianRank",
    "InverseMedianRank",
    #
    "HitsAtK",
    "AdjustedHitsAtK",
    "ZHitsAtK",
    #
    "StandardDeviation",
    "Variance",
    "Count",
    # Misc
    "NoClosedFormError",
    "generate_ranks",
    "generate_num_candidates_and_ranks",
    "generalized_harmonic_numbers",
    "AffineTransformationParameters",
    "harmonic_variances",
    #
    "HITS_METRICS",
]

EPSILON = 1.0e-12


def generate_ranks(
    num_candidates: np.ndarray,
    prefix_shape: Tuple[int, ...] = tuple(),
    seed: Union[None, int, np.random.Generator] = None,
    dtype: Optional[Type[np.number]] = None,
) -> np.ndarray:
    """
    Generate random ranks from a given array of the number of candidates for each ranking task.

    :param num_candidates: shape: s
        the number of candidates
    :param prefix_shape:
        additional dimensions for broadcasted sampling
    :param seed:
        the random seed
    :param dtype:
        the data type

    :return: shape: dims + s
        an array of sampled rank values
    """
    if dtype is None:
        dtype = int
    generator = np.random.default_rng(seed=seed)
    return generator.integers(low=1, high=num_candidates + 1, size=prefix_shape + num_candidates.shape, dtype=dtype)


def generate_num_candidates_and_ranks(
    num_ranks: int,
    max_num_candidates: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random number of candidates, and coherent ranks.

    :param num_ranks:
        the number of ranks to generate
    :param max_num_candidates:
        the maximum number of candidates (e.g., the number of entities)
    :param seed:
        the random seed.

    :return: shape: (num_ranks,)
        a pair of integer arrays, ranks and num_candidates for each individual ranking task
    """
    generator = np.random.default_rng(seed=seed)
    num_candidates = generator.integers(low=1, high=max_num_candidates, size=(num_ranks,))
    ranks = generate_ranks(num_candidates=num_candidates, seed=generator)
    return ranks, num_candidates


class NoClosedFormError(ValueError):
    """The metric does not provide a closed-form implementation for the requested operation."""


class RankBasedMetric(Metric):
    """A base class for rank-based metrics."""

    # rank based metrics do not need binarized scores
    binarize: ClassVar[bool] = False

    #: the supported rank types. Most of the time equal to all rank types
    supported_rank_types: ClassVar[Collection[RankType]] = RANK_TYPES

    #: whether the metric requires the number of candidates for each ranking task
    needs_candidates: ClassVar[bool] = False

    @abstractmethod
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Evaluate the metric.

        :param ranks: shape: s
            the individual ranks
        :param num_candidates: shape: s
            the number of candidates for each individual ranking task
        :param weights: shape: s
            the weights for the individual ranks
        """
        raise NotImplementedError

    def get_sampled_values(
        self,
        num_candidates: np.ndarray,
        num_samples: int,
        weights: Optional[np.ndarray] = None,
        generator: Optional[np.random.Generator] = None,
        memory_intense: bool = True,
    ) -> np.ndarray:
        """
        Calculate the metric on sampled rank arrays.

        :param num_candidates: shape: s
            the number of candidates for each ranking task
        :param num_samples:
            the number of samples
        :param weights: shape: s
            the weights for the individual ranking tasks
        :param generator:
            a random state for reproducibility
        :param memory_intense:
            whether to use a more memory-intense, but more time-efficient variant

        :return: shape: (num_samples,)
            the metric evaluated on `num_samples` sampled rank arrays
        """
        num_candidates = np.asarray(num_candidates)
        if generator is None:
            generator = np.random.default_rng()
        if memory_intense:
            return np.apply_along_axis(
                self,
                axis=1,
                arr=generate_ranks(prefix_shape=(num_samples,), num_candidates=num_candidates, seed=generator),
                num_candidates=num_candidates,
                weights=weights,
            )
        return np.asanyarray(
            a=[
                self(
                    ranks=generate_ranks(num_candidates=num_candidates, seed=generator),
                    num_candidates=num_candidates,
                    weights=weights,
                )
                for _ in range(num_samples)
            ]
        )

    def _bootstrap(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        num_candidates: np.ndarray,
        num_samples: int,
        confidence_level: float = 95.0,
        n_boot: int = 1_000,
        generator: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> np.ndarray:
        """Bootstrap a metric's confidence intervals."""
        # normalize confidence level
        if not (50 < confidence_level < 100):
            raise ValueError(f"Invalid confidence_level={confidence_level}. Should be in (50, 100).")
        p = 50 - confidence_level / 2, 50, 50 + confidence_level / 2

        # sample metric values
        generator = np.random.default_rng(generator)
        xs = self.get_sampled_values(
            num_candidates=num_candidates, num_samples=num_samples, generator=generator, **kwargs
        )

        # bootstrap estimator (i.e., compute on sample with replacement)
        n = xs.shape[0]
        vs = np.asanyarray([func(xs[generator.integers(n, size=(n,))]) for _ in range(n_boot)])
        return np.percentile(vs, p)

    def numeric_expected_value(self, **kwargs) -> float:
        r"""
        Compute expected metric value by summation.

        The expectation is computed under the assumption that each individual rank follows a discrete uniform
        distribution $\mathcal{U}\left(1, N_i\right)$, where $N_i$ denotes the number of candidates for
        ranking task $r_i$.

        :param kwargs:
            keyword-based parameters passed to :func:`get_sampled_values`

        :return:
            The estimated expected value of this metric

        .. warning ::

            Depending on the metric, the estimate may not be very accurate and converge slowly, cf.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.expect.html
        """
        return self.get_sampled_values(**kwargs).mean().item()

    def numeric_expected_value_with_ci(self, **kwargs) -> np.ndarray:
        """Estimate expected value with confidence intervals."""
        return self._bootstrap(func=np.mean, **kwargs)

    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        r"""Compute expected metric value.

        The expectation is computed under the assumption that each individual rank follows a
        discrete uniform distribution $\mathcal{U}\left(1, N_i\right)$, where $N_i$ denotes
        the number of candidates for ranking task $r_i$.

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation, if no closed form
            expected value is implemented
        :param weights: shape: s
            the weights for the individual ranking tasks
        :param kwargs:
            additional keyword-based parameters passed to :func:`get_sampled_values`,
            if no closed form solution is available

        :return:
            the expected value of this metric

        :raises NoClosedFormError:
            raised if a closed form expectation has not been implemented and no number of samples are given

        .. note::
            Prefers analytical solution, if available, but falls back to numeric
            estimation via summation, cf. :func:`RankBasedMetric.numeric_expected_value`.
        """
        if num_samples is None:
            raise NoClosedFormError("Numeric estimation requires to specify a number of samples.")
        return self.numeric_expected_value(
            num_candidates=num_candidates, num_samples=num_samples, weights=weights, **kwargs
        )

    def numeric_variance(self, **kwargs) -> float:
        r"""Compute variance by summation.

        The variance is computed under the assumption that each individual rank follows a discrete uniform
        distribution $\mathcal{U}\left(1, N_i\right)$, where $N_i$ denotes the number of candidates for
        ranking task $r_i$.

        :param kwargs:
            keyword-based parameters passed to :func:`get_sampled_values`

        :return:
            The estimated variance of this metric

        .. warning ::

            Depending on the metric, the estimate may not be very accurate and converge slowly, cf.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.expect.html
        """
        return self.get_sampled_values(**kwargs).var(ddof=1).item()

    def numeric_variance_with_ci(self, **kwargs) -> np.ndarray:
        """Estimate variance with confidence intervals."""
        return self._bootstrap(func=np.var, **kwargs)

    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        r"""Compute variance.

        The variance is computed under the assumption that each individual rank follows a discrete uniform
        distribution $\mathcal{U}\left(1, N_i\right)$, where $N_i$ denotes the number of candidates for
        ranking task $r_i$.

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation, if no closed form
            expected value is implemented
        :param weights: shape: s
            the weights for the individual ranking tasks
        :param kwargs:
            additional keyword-based parameters passed to :func:`get_sampled_values`,
            if no closed form solution is available

        :return:
            The variance of this metric

        :raises NoClosedFormError:
            Raised if a closed form variance has not been implemented and no
            number of samples are given

        .. note::
            Prefers analytical solution, if available, but falls back to numeric
            estimation via summation, cf. :func:`RankBasedMetric.numeric_variance`.
        """
        if num_samples is None:
            raise NoClosedFormError("Numeric estimation requires to specify a number of samples.")
        return self.numeric_variance(num_candidates=num_candidates, num_samples=num_samples, weights=weights, **kwargs)

    def std(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Compute the standard deviation.

        :param num_candidates:
            the number of candidates for each individual rank computation
        :param num_samples:
            the number of samples to use for simulation, if no closed form
            expected value is implemented
        :param weights: shape: s
            the weights for the individual ranking tasks
        :param kwargs:
            additional keyword-based parameters passed to :func:`variance`,

        :return:
            The standard deviation (i.e. the square root of the variance) of this metric

        For a detailed explanation, cf. :func:`RankBasedMetric.variance`.
        """
        return math.sqrt(
            self.variance(num_candidates=num_candidates, num_samples=num_samples, weights=weights, **kwargs)
        )


def _safe_divide(x: float, y: float) -> float:
    """Divide x by y making sure that abs(y) > epsilon."""
    # cf. https://stackoverflow.com/questions/1986152/why-doesnt-python-have-a-sign-function
    y_sign = math.copysign(1.0, y)
    y_abs = abs(y)
    y_abs = max(y_abs, EPSILON)
    y = y_abs * y_sign
    return x / y


class AffineTransformationParameters(NamedTuple):
    """The parameters of an affine transformation."""

    scale: float = 1.0
    offset: float = 0.0


class DerivedRankBasedMetric(RankBasedMetric, ABC):
    r"""
    A derived rank-based metric.

    The derivation is based on an affine transformation of the metric, where scale and bias may depend on the number
    of candidates. Since the transformation only depends on the number of candidates, but not the ranks of the
    predictions, this method can also be used to adjust published results without access to the trained models.
    Moreover, we can obtain closed form solutions for expected value and variance.

    Let $\alpha, \beta$ denote the scale and offset of the affine transformation, i.e.,

    .. math ::

        M^* = \alpha \cdot M + \beta

    Then we have for the expectation

    .. math ::

        \mathbb{E}[M^*] = \mathbb{E}[\alpha \cdot M + \beta]
                        = \alpha \cdot \mathbb{E}[M] + \beta

    and for the variance

    .. math ::

        \mathbb{V}[M^*] = \mathbb{V}[\alpha \cdot M + \beta]
                        = \alpha^2 \cdot \mathbb{V}[M]
    """

    base: RankBasedMetric
    needs_candidates: ClassVar[bool] = True

    #: The rank-based metric class that this derived metric extends
    base_cls: ClassVar[Optional[Type[RankBasedMetric]]] = None

    def __init__(
        self,
        base_cls: HintOrType[RankBasedMetric] = None,
        **kwargs,
    ):
        """
        Initialize the derived metric.

        :param base_cls:
            the base class, or a hint thereof. If None, use the class-attribute
        :param kwargs:
            additional keyword-based parameters used to instantiate the base metric
        """
        self.base = rank_based_metric_resolver.make(base_cls or self.base_cls, pos_kwargs=kwargs)

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        if num_candidates is None:
            raise ValueError(f"{self.__class__.__name__} requires number of candidates.")
        return self.adjust(
            base_metric_result=self.base(ranks=ranks, num_candidates=num_candidates, weights=weights),
            num_candidates=num_candidates,
            weights=weights,
        )

    def adjust(
        self, base_metric_result: float, num_candidates: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Adjust base metric results based on the number of candidates.

        :param base_metric_result:
            the result of the base metric
        :param num_candidates:
            the number of candidates
        :param weights: shape: s
            the weights for the individual ranking tasks

        :return:
            the adjusted metric

        .. note ::

            since the adjustment only depends on the number of candidates, but not the ranks of the predictions, this
            method can also be used to adjust published results without access to the trained models.
        """
        parameters = self.get_coefficients(num_candidates=num_candidates, weights=weights)
        return parameters.scale * base_metric_result + parameters.offset

    # docstr-coverage: inherited
    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        # since scale and offset are constant for a given number of candidates, we have
        # E[scale * M + offset] = scale * E[M] + offset
        return self.adjust(
            base_metric_result=self.base.expected_value(
                num_candidates=num_candidates, num_samples=num_samples, weights=weights, **kwargs
            ),
            num_candidates=num_candidates,
            weights=weights,
        )

    # docstr-coverage: inherited
    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        # since scale and offset are constant for a given number of candidates, we have
        # V[scale * M + offset] = scale^2 * V[M]
        parameters = self.get_coefficients(num_candidates=num_candidates, weights=weights)
        return parameters.scale**2.0 * self.base.variance(
            num_candidates=num_candidates, num_samples=num_samples, weights=weights, **kwargs
        )

    @abstractmethod
    def get_coefficients(
        self, num_candidates: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> AffineTransformationParameters:
        """
        Compute the scaling coefficients.

        :param num_candidates:
            the number of candidates
        :param weights:
            the weights for the individual ranking tasks


        :return:
            a tuple (scale, offset)
        """
        raise NotImplementedError


class ZMetric(DerivedRankBasedMetric):
    r"""
    A z-score adjusted metrics.

    .. math ::

        \mathbb{M}^* = \frac{\mathbb{M} - \mathbb{E}[\mathbb{M}]}{\sqrt{\mathbb{V}[\mathbb{M}]}}

    In terms of the affine transformation from DerivedRankBasedMetric, we obtain the following coefficients:

    .. math ::

        \alpha &= \frac{1}{\sqrt{\mathbb{V}[\mathbb{M}]}} \\
        \beta  &= -\alpha \cdot \mathbb{E}[\mathbb{M}]

    .. note ::

        For non-increasing metrics, i.e., where larger values correspond to better results, we additionally change the
        sign of the result such that a larger z-value always corresponds to a better result irrespective of the base
        metric's direction.

    .. warning:: This requires a closed-form solution to the expected value and the variance
    """

    #: Z-adjusted metrics are formulated to be increasing
    increasing = True
    #: Z-adjusted metrics can only be applied to realistic ranks
    supported_rank_types = (RANK_REALISTIC,)
    value_range = ValueRange(lower=None, upper=None)
    closed_expectation: ClassVar[bool] = True
    closed_variance: ClassVar[bool] = True

    # docstr-coverage: inherited
    def get_coefficients(
        self, num_candidates: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> AffineTransformationParameters:  # noqa: D102
        mean = self.base.expected_value(num_candidates=num_candidates, weights=weights)
        std = self.base.std(num_candidates=num_candidates, weights=weights)
        scale = _safe_divide(1.0, std)
        if not self.base.increasing:
            scale = -scale
        offset = -scale * mean
        return AffineTransformationParameters(scale=scale, offset=offset)

    # docstr-coverage: inherited
    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        # should be exactly 0.0
        return 0.0  # centered

    # docstr-coverage: inherited
    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        # should be exactly 1.0
        return 1.0  # re-scaled


class ExpectationNormalizedMetric(DerivedRankBasedMetric):
    r"""An adjustment to create an expectation-normalized metric.

    .. math ::

        M^* = \frac{M}{\mathbb{E}[M]}

    In terms of the affine transformation from :class:`DerivedRankBasedMetric`, we obtain the following coefficients:

    .. math ::

        \alpha &= \frac{1}{\mathbb{E}[M]} \\
        \beta  &= 0

    .. warning:: This requires a closed-form solution to the expected value
    """

    closed_expectation: ClassVar[bool] = True

    # docstr-coverage: inherited
    def get_coefficients(
        self, num_candidates: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> AffineTransformationParameters:  # noqa: D102
        return AffineTransformationParameters(
            scale=_safe_divide(1, self.base.expected_value(num_candidates=num_candidates, weights=weights))
        )

    # docstr-coverage: inherited
    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        return 1.0  # centered


class ReindexedMetric(DerivedRankBasedMetric):
    r"""A mixin to create an expectation normalized metric with max of 1 and expectation of 0.

    .. math::

        \mathbb{M}^{*} = \frac{\mathbb{M} - \mathbb{E}[\mathbb{M}]}{1 - \mathbb{E}[\mathbb{M}]}

    In terms of the affine transformation from DerivedRankBasedMetric, we obtain the following coefficients:

    .. math ::

        \alpha &= \frac{1}{1 - \mathbb{E}[\mathbb{M}]} \\
        \beta  &= -\alpha \cdot \mathbb{E}[\mathbb{M}]

    .. warning:: This requires a closed-form solution to the expected value
    """

    #: Expectation/maximum reindexed metrics are formulated to be increasing
    increasing = True
    #: Expectation/maximum reindexed metrics can only be applied to realistic ranks
    supported_rank_types = (RANK_REALISTIC,)
    closed_expectation: ClassVar[bool] = True

    # docstr-coverage: inherited
    def get_coefficients(
        self, num_candidates: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> AffineTransformationParameters:  # noqa: D102
        mean = self.base.expected_value(num_candidates=num_candidates, weights=weights)
        scale = _safe_divide(1.0, 1.0 - mean)
        offset = -scale * mean
        return AffineTransformationParameters(scale=scale, offset=offset)

    # docstr-coverage: inherited
    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        # should be exactly 0.0
        return 0.0


@parse_docdata
class ArithmeticMeanRank(RankBasedMetric):
    r"""The (arithmetic) mean rank.

    The mean rank (MR) computes the arithmetic mean over all individual ranks.
    Denoting the set of individual ranks as $\mathcal{I}$, it is given as:

    .. math::

        MR =\frac{1}{|\mathcal{I}|} \sum \limits_{r \in \mathcal{I}} r

    It has the advantage over hits @ k that it is sensitive to any model performance changes, not only what occurs
    under a certain cutoff and therefore reflects average performance. With PyKEEN's standard 1-based indexing,
    the mean rank lies on the interval $[1, \infty)$ where lower is better.

    .. warning::

        While the arithmetic mean rank is interpretable, the mean rank is dependent on the number of candidates.
        A mean rank of 10 might indicate strong performance for a candidate set size of 1,000,000,
        but incredibly poor performance for a candidate set size of 20.

    For the expected value, we have

    .. math::

        \mathbb{E}[MR] &= \mathbb{E}[\frac{1}{n} \sum \limits_{i=1}^{n} r_i] \\
                       &= \frac{1}{n} \sum \limits_{i=1}^{n} \mathbb{E}[r_i] \\
                       &= \frac{1}{n} \sum \limits_{i=1}^{n} \frac{N_i + 1}{2}

    For the variance, we have

    .. math::

        \mathbb{V}[MR] &= \mathbb{V}[\frac{1}{n} \sum \limits_{i=1}^{n} r_i] \\
                       &= \frac{1}{n^2} \sum \limits_{i=1}^{n} \mathbb{V}[r_i] \\
                       &= \frac{1}{n^2} \sum \limits_{i=1}^{n} \frac{N_i^2 - 1}{12} \\
                       &= \frac{1}{12 n^2} \cdot \left(-n + \sum \limits_{i=1}^{n} N_i \right)

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html#mean-rank
    description: The arithmetic mean over all ranks.
    """

    name = "Mean Rank (MR)"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing: ClassVar[bool] = False
    synonyms: ClassVar[Collection[str]] = ("mean_rank", "mr")
    supports_weights: ClassVar[bool] = True
    closed_expectation: ClassVar[bool] = True
    closed_variance: ClassVar[bool] = True

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.average(np.asanyarray(ranks), weights=weights).item()

    # docstr-coverage: inherited
    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        num_candidates = np.asanyarray(num_candidates)
        individual_expectation = 0.5 * (num_candidates + 1)
        return weighted_mean_expectation(individual=individual_expectation, weights=weights)

    # docstr-coverage: inherited
    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        num_candidates = np.asanyarray(num_candidates)
        individual_variance = (num_candidates**2 - 1) / 12.0
        return weighted_mean_variance(individual=individual_variance, weights=weights)


@parse_docdata
class ZArithmeticMeanRank(ZMetric):
    """The z-scored arithmetic mean rank.

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The z-scored mean rank
    """

    name = "z-Mean Rank (zMR)"
    synonyms: ClassVar[Collection[str]] = ("zamr", "zmr")
    base_cls = ArithmeticMeanRank
    supports_weights: ClassVar[bool] = ArithmeticMeanRank.supports_weights


@parse_docdata
class InverseArithmeticMeanRank(RankBasedMetric):
    """The inverse arithmetic mean rank.

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The inverse of the arithmetic mean over all ranks.
    """

    name = "Inverse Arithmetic Mean Rank (IAMR)"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True
    synonyms: ClassVar[Collection[str]] = ("iamr",)
    supports_weights = True

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.reciprocal(np.average(np.asanyarray(ranks), weights=weights)).item()


@parse_docdata
class GeometricMeanRank(RankBasedMetric):
    r"""The (weighted) geometric mean rank.

    It is given by

    .. math::

        M = \left(\prod \limits_{i=1}^{m} r_i^{w_i}\right)^{1/w}

    with $w = \sum \limits_{i=1}^{m} w_i$. The unweighted GMR is obtained by setting $w_i = 1$.

    For computing the expected value, we first observe that

    .. math::

        \mathbb{E}[M] &= \mathbb{E}\left[\sqrt[w]{\prod \limits_{i=1}^{m} r_i^{w_i}}\right] \\
                      &= \prod \limits_{i=1}^{m} \mathbb{E}[r_i^{w_i/w}] \\
                      &= \exp \sum \limits_{i=1}^{m} \log \mathbb{E}[r_i^{w_i/w}]

    where the last steps permits a numerically more stable computation. Moreover, we have

    .. math::

        \log \mathbb{E}[r_i^{w_i/w}]
            &= \log \frac{1}{N_i} \sum \limits_{j=1}^{N_i} j^{w_i/w} \\
            &= -\log \frac{1}{N_i} + \log \sum \limits_{j=1}^{N_i} j^{w_i/w} \\
            &= -\log \frac{1}{N_i} + \log \sum \limits_{j=1}^{N_i} \exp \log j^{w_i/w} \\
            &= -\log \frac{1}{N_i} + \log \sum \limits_{j=1}^{N_i} \exp ( \frac{w_i}{w} \cdot \log j )

    For the second summand in the last line, we observe a log-sum-exp term, with known numerically stable
    implementation.

    Alternatively, we can write

    .. math::
        \log \mathbb{E}[r_i^{w_i/w}]
            &= \log \frac{1}{N_i} \sum \limits_{j=1}^{N_i} j^{w_i/w} \\
            &= \log \frac{H_{-w_i/w}(N_i)}{N_i} \\
            &= \log H_{-w_i/w}(N_i) - \log N_i

    .. math::
        \mathbb{E}[M]
            &= \exp \sum \limits_{i=1}^{m} \log \mathbb{E}[r_i^{w_i/w}] \\
            &= \exp \sum \limits_{i=1}^{m} (\log H_{-w_i/w}(N_i) - \log N_i) \\
            &= \exp \sum \limits_{i=1}^{m} \log H_{-w_i/w}(N_i) - \exp \sum \limits_{i=1}^{m} \log N_i

    where $H_p(n)$ denotes the generalized harmonic number, cf. :func:`generalized_harmonic_numbers`.
    ---
    link: https://arxiv.org/abs/2203.07544
    description: The geometric mean over all ranks.
    """

    name = "Geometric Mean Rank (GMR)"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("gmr",)
    supports_weights = True
    closed_expectation: ClassVar[bool] = True
    closed_variance: ClassVar[bool] = True

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return stats.gmean(ranks, weights=weights).item()

    # docstr-coverage: inherited
    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        is_log, individual = self._individual_expectation(num_candidates=num_candidates, weights=weights)
        return stable_product(individual, is_log=is_log).item()

    # docstr-coverage: inherited
    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        # V (prod x_i) = prod (V[x_i] - E[x_i]^2) - prod(E[x_i])^2
        is_log, individual_expectation = self._individual_expectation(num_candidates=num_candidates, weights=weights)
        if is_log:
            individual_expectation = np.exp(individual_expectation)
        individual_variance = self._individual_variance(
            num_candidates=num_candidates, weights=weights, individual_expectation=individual_expectation
        )
        return (
            stable_product(individual_variance + individual_expectation**2)
            - stable_product(individual_expectation) ** 2
        )

    @classmethod
    def _individual_variance(
        cls, num_candidates: np.ndarray, weights: np.ndarray, individual_expectation: np.ndarray
    ) -> np.ndarray:
        # use V[x] = E[x^2] - E[x]^2
        x2 = (
            np.exp(cls._log_individual_expectation_no_weight(num_candidates=num_candidates, factor=2.0))
            if weights is None
            else cls._individual_expectation_weighted(num_candidates=num_candidates, weights=weights, factor=2.0)
        )
        return x2 - individual_expectation**2

    @classmethod
    def _individual_expectation(
        cls, num_candidates: np.ndarray, weights: Optional[np.ndarray]
    ) -> Tuple[bool, np.ndarray]:
        if weights is None:
            return True, cls._log_individual_expectation_no_weight(num_candidates=num_candidates)
        return False, cls._individual_expectation_weighted(num_candidates=num_candidates, weights=weights)

    @staticmethod
    def _individual_expectation_weighted(
        num_candidates: np.ndarray, weights: np.ndarray, factor: float = 1.0
    ) -> np.ndarray:
        weights = factor * weights / weights.sum()
        x = np.empty_like(weights)
        # group by same weight -> compute H_w(n) for multiple n at once
        unique_weights, inverse = np.unique(weights, return_inverse=True)
        for i, w in enumerate(unique_weights):
            mask = inverse == i
            nc = num_candidates[mask]
            h = generalized_harmonic_numbers(nc.max(), p=w)
            x[mask] = h[nc - 1] / nc
        return x

    @staticmethod
    def _log_individual_expectation_no_weight(num_candidates: np.ndarray, factor: float = 1.0) -> np.ndarray:
        m = num_candidates.size
        # we compute log E[r_i^(1/m)] for all N_i = 1 ... max_N_i once
        max_val = num_candidates.max()
        x = np.arange(1, max_val + 1, dtype=float)
        x = factor * np.log(x) / m
        x = logcumsumexp(x)
        # now select from precomputed cumulative sums and aggregate
        x = x[num_candidates - 1] - np.log(num_candidates)
        return x


@parse_docdata
class InverseGeometricMeanRank(RankBasedMetric):
    r"""The inverse geometric mean rank.

    The mean rank corresponds to the arithmetic mean, and tends to be more affected by high rank values.
    The mean reciprocal rank corresponds to the harmonic mean, and tends to be more affected by low rank values.
    The remaining Pythagorean mean, the geometric mean, lies in the center and therefore could better balance these
    biases. Therefore, the inverse geometric mean rank (IGMR) is defined as:

    .. math::

        IGMR = \sqrt[\|\mathcal{I}\|]{\prod \limits_{r \in \mathcal{I}} r}

    .. note:: This metric is novel as of its implementation in PyKEEN and was proposed by Max Berrendorf

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The inverse of the geometric mean over all ranks.
    """

    name = "Inverse Geometric Mean Rank (IGMR)"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True
    synonyms: ClassVar[Collection[str]] = ("igmr",)
    supports_weights = True

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.reciprocal(stats.gmean(ranks, weights=weights)).item()


@parse_docdata
class HarmonicMeanRank(RankBasedMetric):
    """The harmonic mean rank.

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The harmonic mean over all ranks.
    """

    name = "Harmonic Mean Rank (HMR)"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("hmr",)
    supports_weights = True

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return weighted_harmonic_mean(a=ranks, weights=weights).item()


def generalized_harmonic_numbers(n: int, p: float = -1.0) -> np.ndarray:
    r"""
    Calculate the generalized harmonic numbers from 1 to n (both inclusive).

    .. math::

        H_p(n) = \sum \limits_{i=1}^{n} i^{-p}

    :param n:
        the maximum number for which the generalized harmonic numbers are calculated
    :param p:
        the power, typically negative

    :return: shape: (n,)
        the first $n$ generalized harmonic numbers

    .. seealso::
        https://en.wikipedia.org/wiki/Harmonic_number#Generalizations
    """
    return np.cumsum(np.power(np.arange(1, n + 1, dtype=float), p))


def harmonic_variances(n: int) -> np.ndarray:
    r"""
    Pre-calculate variances of inverse rank distributions.

    With

    .. math::

        H_p(n) = \sum \limits_{i=1}^{n} i^{-p}

    denoting the generalized harmonic numbers, and abbreviating $H(n) := H_1(n)$, we have

    .. math::

        \textit{V}[n]
            &= \frac{1}{n} \sum \limits_{i=1}^n \left( i^{-1} - \frac{H(n)}{n} \right)^2 \\
            &= \frac{n \cdot H_2(n) - H(n)^2}{n^2}

    :param n:
        the maximum rank number

    :return: shape: (n+1,)
        the variances for the discrete uniform distribution over $\{\frac{1}{1}, \dots, \frac{1}{k}\}$`
    """
    h = generalized_harmonic_numbers(n)
    h2 = generalized_harmonic_numbers(n, p=-2)
    n = np.arange(1, n + 1, dtype=float)
    v = (n * h2 - h**2) / n**2
    # ensure non-negativity, mathematically not necessary, but just to be safe from the numeric perspective
    # cf. https://en.wikipedia.org/wiki/Loss_of_significance#Subtraction
    v = np.maximum(v, 0.0)
    return v


@parse_docdata
class InverseHarmonicMeanRank(RankBasedMetric):
    r"""The inverse harmonic mean rank.

    The mean reciprocal rank (MRR) is the arithmetic mean of reciprocal ranks, and thus the inverse of the harmonic mean
    of the ranks. It is defined as:

    .. math::

        IHMR = MRR =\frac{1}{|\mathcal{I}|} \sum_{r \in \mathcal{I}} r^{-1}

    .. warning::

        It has been argued that the mean reciprocal rank has theoretical flaws by [fuhr2018]_. However, this opinion
        is not undisputed, cf. [sakai2021]_.

    Despite its flaws, MRR is still often used during early stopping due to its behavior related to low rank values.
    While the hits @ k ignores changes among high rank values completely and the mean rank changes uniformly
    across the full value range, the mean reciprocal rank is more affected by changes of low rank values than high ones
    (without disregarding them completely like hits @ k does for low rank values)
    Therefore, it can be considered as soft a version of hits @ k that is less sensitive to outliers.
    It is bound on $(0, 1]$ where closer to 1 is better.

    Let

    .. math::

        H_m(n) = \sum \limits_{i=1}^{n} i^{-m}

    denote the generalized harmonic number, with $H(n) := H_{1}(n)$ for brevity.
    Thus, we have

    .. math::

        \mathbb{E}\left[r_i^{-1}\right] = \frac{H(N_i)}{N_i}

    and hence

    .. math::

        \mathbb{E}\left[\textrm{MRR}\right]
            &= \mathbb{E}\left[\frac{1}{n} \sum \limits_{i=1}^n r_i^{-1}\right] \\
            &= \frac{1}{n} \sum \limits_{i=1}^n \mathbb{E}\left[r_i^{-1}\right] \\
            &= \frac{1}{n} \sum \limits_{i=1}^n \frac{H(N_i)}{N_i}

    For the variance, we have for the individual ranks

    .. math::

        \mathbb{V}\left[r_i^{-1}\right]
            &= \frac{1}{N_i} \sum \limits_{i=1}^{N_i} \left(\frac{H(N_i)}{N_i} - \frac{1}{i}\right)^2 \\
            &= \frac{N_i \cdot H_2(N_i) - H(N_i)^2}{N_i^2}

    and thus overall

    .. math::

        \mathbb{V}\left[\textrm{MRR}\right]
            &= \mathbb{V}\left[\frac{1}{n} \sum \limits_{i=1}^n r_i^{-1}\right] \\
            &= \frac{1}{n^2} \sum \limits_{i=1}^n \mathbb{V}\left[r_i^{-1}\right] \\
            &= \frac{1}{n^2} \sum \limits_{i=1}^n \frac{N_i \cdot H_2(N_i) - H(N_i)^2}{N_i^2} \\

    .. seealso::
        https://en.wikipedia.org/wiki/Inverse_distribution#Inverse_uniform_distribution

    ---
    link: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    description: The inverse of the harmonic mean over all ranks.
    """

    name = "Mean Reciprocal Rank (MRR)"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    synonyms: ClassVar[Collection[str]] = ("mean_reciprocal_rank", "mrr")
    increasing = True
    supports_weights = True
    closed_expectation: ClassVar[bool] = True
    closed_variance: ClassVar[bool] = True

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.reciprocal(weighted_harmonic_mean(a=ranks, weights=weights)).item()

    # docstr-coverage: inherited
    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        num_candidates = np.asanyarray(num_candidates)
        n = num_candidates.max().item()
        expectation = generalized_harmonic_numbers(n, p=-1.0) / np.arange(1, n + 1)
        individual = expectation[num_candidates - 1]
        return weighted_mean_expectation(individual, weights)

    # docstr-coverage: inherited
    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        num_candidates = np.asanyarray(num_candidates)
        n = num_candidates.max().item()
        individual = harmonic_variances(n)[num_candidates - 1]
        return weighted_mean_variance(individual, weights)


@parse_docdata
class AdjustedInverseHarmonicMeanRank(ReindexedMetric):
    r"""The adjusted MRR index.

    .. note ::
        the actual lower bound is $\frac{-\mathbb{E}[\text{MRR}]}{1-\mathbb{E}[\text{MRR}]}$,
        and thus data dependent.

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The re-indexed adjusted MRR
    tight_lower: -E[f]/(1-E[f])
    """

    name = "Adjusted Inverse Harmonic Mean Rank"
    synonyms: ClassVar[Collection[str]] = ("amrr", "aihmr", "adjusted_mrr", "adjusted_mean_reciprocal_rank")
    value_range = ValueRange(lower=None, lower_inclusive=False, upper=1, upper_inclusive=True)
    base_cls = InverseHarmonicMeanRank
    supports_weights: ClassVar[bool] = InverseHarmonicMeanRank.supports_weights


@parse_docdata
class ZInverseHarmonicMeanRank(ZMetric):
    """The z-inverse harmonic mean rank (ZIHMR).

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The z-scored mean reciprocal rank
    """

    name = "z-Mean Reciprocal Rank (zMRR)"
    synonyms: ClassVar[Collection[str]] = ("zmrr", "zihmr")
    base_cls = InverseHarmonicMeanRank
    supports_weights: ClassVar[bool] = InverseHarmonicMeanRank.supports_weights


@parse_docdata
class ZGeometricMeanRank(ZMetric):
    """The z geometric mean rank (zGMR).

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The z-scored geometric mean rank
    """

    name = "z-Geometric Mean Rank (zGMR)"
    synonyms: ClassVar[Collection[str]] = ("zgmr",)
    base_cls = GeometricMeanRank
    supports_weights: ClassVar[bool] = GeometricMeanRank.supports_weights


@parse_docdata
class MedianRank(RankBasedMetric):
    """The median rank.

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The median over all ranks.
    """

    name = "Median Rank"
    value_range = ValueRange(lower=1, lower_inclusive=True, upper=math.inf)
    increasing = False
    supports_weights = True

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        if weights is None:
            return np.median(ranks).item()

        return weighted_median(a=ranks, weights=weights).item()


@parse_docdata
class InverseMedianRank(RankBasedMetric):
    """The inverse median rank.

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The inverse of the median over all ranks.
    """

    name = "Inverse Median Rank"
    value_range = ValueRange(lower=0, lower_inclusive=False, upper=1, upper_inclusive=True)
    increasing = True
    supports_weights = True

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.reciprocal(weighted_median(a=ranks, weights=weights)).item()


@parse_docdata
class StandardDeviation(RankBasedMetric):
    """The ranks' standard deviation.

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html
    """

    name = "Standard Deviation (std)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("rank_std", "std")

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.asanyarray(ranks).std().item()


@parse_docdata
class Variance(RankBasedMetric):
    """The ranks' variance.

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html
    """

    name = "Variance"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("rank_var", "var")

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.asanyarray(ranks).var().item()


@parse_docdata
class MedianAbsoluteDeviation(RankBasedMetric):
    """The ranks' median absolute deviation (MAD).

    ---
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html
    """

    name = "Median Absolute Deviation (MAD)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("rank_mad", "mad")
    supports_weights = True

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        if weights is None:
            return stats.median_abs_deviation(ranks, scale="normal").item()

        return weighted_median(a=np.abs(ranks - weighted_median(a=ranks, weights=weights)), weights=weights).item()


@parse_docdata
class Count(RankBasedMetric):
    """The ranks' count.

    Lower numbers may indicate unreliable results.
    ---
    link: https://pykeen.readthedocs.io/en/stable/reference/evaluation.html
    """

    name = "Count"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=math.inf)
    increasing = False
    synonyms: ClassVar[Collection[str]] = ("rank_count",)

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        # TODO: should we return the sum of weights?
        return float(np.asanyarray(ranks).size)


@parse_docdata
class HitsAtK(RankBasedMetric):
    r"""The Hits @ k.

    The hits @ k describes the fraction of true entities that appear in the first $k$ entities of the sorted rank list.
    Denoting the set of individual ranks as $\mathcal{I}$, it is given as:

    .. math::

        H_k = \frac{1}{|\mathcal{I}|} \sum \limits_{r \in \mathcal{I}} \mathbb{I}[r \leq k]

    For example, if Google shows 20 results on the first page, then the percentage of results that are relevant is the
    hits @ 20. The hits @ k, regardless of $k$, lies on the $[0, 1]$ where closer to 1 is better.

    .. warning::

        This metric does not differentiate between cases when the rank is larger than $k$.
        This means that a miss with rank $k+1$ and $k+d$ where $d \gg 1$ have the same
        effect on the final score. Therefore, it is less suitable for the comparison of different
        models.

    For the expected values, we first note that

    .. math::

        \mathbb{I}[r_i \leq k] \sim \textit{Bernoulli}(p_i)

    with $p_i = \min\{\frac{k}{N_i}, 1\}$. Thus, we have

    .. math::

        \mathbb{E}[\mathbb{I}[r_i \leq k]] = p_i

    and

    .. math::

        \mathbb{V}[\mathbb{I}[r_i \leq k]] = p_i \cdot (1 - p_i)

    Hence, we obtain

    .. math::

        \mathbb{E}[Hits@k] &= \mathbb{E}\left[\frac{1}{n} \sum \limits_{i=1}^{n} \mathbb{I}[r_i \leq k]\right] \\
                           &= \frac{1}{n} \sum \limits_{i=1}^{n} \mathbb{E}[\mathbb{I}[r_i \leq k]] \\
                           &= \frac{1}{n} \sum \limits_{i=1}^{n} p_i

    For the variance, we have

    .. math::

        \mathbb{V}[Hits@k] &= \mathbb{V}\left[\frac{1}{n} \sum \limits_{i=1}^{n} \mathbb{I}[r_i \leq k]\right] \\
                           &= \frac{1}{n^2} \sum \limits_{i=1}^{n} \mathbb{V}\left[\mathbb{I}[r_i \leq k]\right] \\
                           &= \frac{1}{n^2} \sum \limits_{i=1}^{n} p_i(1 - p_i)
    ---
    description: The relative frequency of ranks not larger than a given k.
    link: https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html#hits-k
    """

    name = "Hits @ K"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=1, upper_inclusive=True)
    synonyms: ClassVar[Collection[str]] = ("h@k", "hits@k", "h@", "hits@", "hits_at_", "h_at_")
    increasing = True
    supports_weights = True
    closed_expectation: ClassVar[bool] = True
    closed_variance: ClassVar[bool] = True

    def __init__(self, k: int = 10) -> None:
        """
        Initialize the metric.

        :param k:
            the parameter $k$ of number of top entries to consider
        """
        super().__init__()
        self.k = k

    # docstr-coverage: inherited
    def _extra_repr(self) -> Iterable[str]:
        yield f"k={self.k}"

    # docstr-coverage: inherited
    def __call__(
        self, ranks: np.ndarray, num_candidates: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None
    ) -> float:  # noqa: D102
        return np.average(np.less_equal(ranks, self.k), weights=weights).item()

    # docstr-coverage: inherited
    @property
    def key(self) -> str:  # noqa: D102
        return super().key[:-1] + str(self.k)

    # docstr-coverage: inherited
    def expected_value(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        num_candidates = np.asanyarray(num_candidates, dtype=float)
        # for each individual ranking task, we have I[r_i <= k] ~ Bernoulli(k/N_i)
        individual = np.minimum(self.k / num_candidates, 1.0)
        return weighted_mean_expectation(individual=individual, weights=weights)

    # docstr-coverage: inherited
    def variance(
        self,
        num_candidates: np.ndarray,
        num_samples: Optional[int] = None,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:  # noqa: D102
        # for each individual ranking task, we have I[r_i <= k] ~ Bernoulli(k/N_i)
        num_candidates = np.asanyarray(num_candidates, dtype=float)
        p = np.minimum(self.k / num_candidates, 1.0)
        individual_variance = p * (1 - p)
        return weighted_mean_variance(individual=individual_variance, weights=weights)


@parse_docdata
class AdjustedHitsAtK(ReindexedMetric):
    r"""The adjusted Hits at K ($AH_k$).

    .. note ::
        the actual lower bound is $\frac{-\mathbb{E}[H_k]}{1 - \mathbb{E}[H_k]}$, and thus data dependent.

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The re-indexed adjusted hits at K
    tight_lower: -E[f]/(1-E[f])
    """

    name = "Adjusted Hits at K"
    synonyms: ClassVar[Collection[str]] = (
        "ahk",
        "ah@k",
        "ahits@k",
        "ah@",
        "ahits@",
        "ahits_at_",
        "ah_at_",
        "adjusted_hits_at_",
    )
    value_range = ValueRange(lower=None, lower_inclusive=False, upper=1, upper_inclusive=True)
    base_cls = HitsAtK
    supports_weights: ClassVar[bool] = HitsAtK.supports_weights


@parse_docdata
class ZHitsAtK(ZMetric):
    """The z-scored hits at k ($ZAH_k$).

    ---
    link: https://arxiv.org/abs/2203.07544
    description: The z-scored hits at K
    """

    name = "z-Hits at K"
    synonyms: ClassVar[Collection[str]] = ("z_hits_at_", "zahk")
    increasing = True
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True
    base_cls = HitsAtK
    supports_weights: ClassVar[bool] = HitsAtK.supports_weights


@parse_docdata
class AdjustedArithmeticMeanRank(ExpectationNormalizedMetric):
    """The adjusted arithmetic mean rank (AMR).

    The adjusted (arithmetic) mean rank (AMR) was introduced by [berrendorf2020]. It is defined as the ratio of the
    mean rank to the expected mean rank. It lies on the open interval $(0, 2)$ where lower is better.

    ---
    description: The mean over all ranks divided by its expected value.
    link: https://arxiv.org/abs/2002.06914
    """

    name = "Adjusted Arithmetic Mean Rank (AAMR)"
    value_range = ValueRange(lower=0, lower_inclusive=True, upper=2, upper_inclusive=False)
    synonyms: ClassVar[Collection[str]] = ("adjusted_mean_rank", "amr", "aamr")
    supported_rank_types = (RANK_REALISTIC,)
    needs_candidates = True
    increasing = False
    base_cls = ArithmeticMeanRank
    supports_weights: ClassVar[bool] = ArithmeticMeanRank.supports_weights


@parse_docdata
class AdjustedArithmeticMeanRankIndex(ReindexedMetric):
    """The adjusted arithmetic mean rank index (AMRI).

    The adjusted (arithmetic) mean rank index (AMRI) was introduced by [berrendorf2020] to make the AMR more intuitive.
    The AMRI has a bounded value range of $[-1, 1]$ where closer to 1 is better.

    ---
    link: https://arxiv.org/abs/2002.06914
    description: The re-indexed adjusted mean rank (AAMR)
    """

    name = "Adjusted Arithmetic Mean Rank Index (AAMRI)"
    value_range = ValueRange(lower=-1, lower_inclusive=True, upper=1, upper_inclusive=True)
    synonyms: ClassVar[Collection[str]] = ("adjusted_mean_rank_index", "amri", "aamri")
    base_cls = ArithmeticMeanRank
    supports_weights: ClassVar[bool] = ArithmeticMeanRank.supports_weights


@parse_docdata
class AdjustedGeometricMeanRankIndex(ReindexedMetric):
    """The adjusted geometric mean rank index (AGMRI).

    ---
    link: https://arxiv.org/abs/2002.06914
    description: The re-indexed adjusted geometric mean rank (AGMRI)
    tight_lower: -E[f]/(1-E[f])
    """

    name = "Adjusted Geometric Mean Rank Index (AGMRI)"
    value_range = ValueRange(lower=None, lower_inclusive=False, upper=1, upper_inclusive=True)
    synonyms: ClassVar[Collection[str]] = ("gmri", "agmri")
    base_cls = GeometricMeanRank
    supports_weights: ClassVar[bool] = GeometricMeanRank.supports_weights


rank_based_metric_resolver: ClassResolver[RankBasedMetric] = ClassResolver.from_subclasses(
    base=RankBasedMetric,
    default=InverseHarmonicMeanRank,  # mrr
    skip={ExpectationNormalizedMetric, ReindexedMetric, ZMetric, DerivedRankBasedMetric},
)
"""The rank-based metric resolver allows for the lookup and instantiation of classes
deriving from :class:`RankBasedMetric` via the :mod:`class_resolver`.
"""

HITS_METRICS: Tuple[Type[RankBasedMetric], ...] = (HitsAtK, ZHitsAtK, AdjustedHitsAtK)
