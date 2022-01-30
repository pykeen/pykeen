"""Expected metric values under random ordering."""

from typing import Sequence, Union

import numpy as np

from .metrics import metric_resolver

__all__ = [
    "expected_mean_rank",
    "expected_hits_at_k",
    "numeric_expected_value",
]


def expected_mean_rank(
    num_candidates: Union[Sequence[int], np.ndarray],
) -> float:
    r"""
    Calculate the expected mean rank under random ordering.

    .. math ::

        E[MR] = \frac{1}{n} \sum \limits_{i=1}^{n} \frac{1 + CSS[i]}{2}
              = \frac{1}{2}(1 + \frac{1}{n} \sum \limits_{i=1}^{n} CSS[i])

    :param num_candidates:
        the number of candidates for each individual rank computation

    :return:
        the expected mean rank
    """
    return 0.5 * (1 + np.mean(np.asanyarray(num_candidates)))


def expected_hits_at_k(
    num_candidates: Union[Sequence[int], np.ndarray],
    k: int,
) -> float:
    r"""
    Calculate the expected Hits@k under random ordering.

    .. math ::

        E[Hits@k] = \frac{1}{n} \sum \limits_{i=1}^{n} min(\frac{k}{CSS[i]}, 1.0)

    :param num_candidates:
        the number of candidates for each individual rank computation

    :return:
        the expected Hits@k value
    """
    return k * np.mean(np.reciprocal(np.asanyarray(num_candidates, dtype=float)).clip(min=None, max=1 / k))


# TODO: closed-forms for other metrics?


def numeric_expected_value(
    metric: str,
    num_candidates: Union[Sequence[int], np.ndarray],
    num_samples: int,
) -> float:
    """
    Compute expected metric value by summation.

    Depending on the metric, the estimate may not be very accurate and converage slowly, cf.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.expect.html
    """
    metric_func = metric_resolver.make(metric)
    num_candidates = np.asarray(num_candidates)
    generator = np.random.default_rng()
    expectation = 0
    for _ in range(num_samples):
        ranks = generator.integers(low=0, high=num_candidates)
        expectation += metric_func(ranks)
    return expectation / num_samples
