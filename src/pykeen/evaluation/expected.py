from typing import Sequence, Union

import numpy as np

__all__ = [
    "expected_mean_rank",
    "expected_hits_at_k",
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
