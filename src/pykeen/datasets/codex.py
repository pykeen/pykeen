# -*- coding: utf-8 -*-

"""CoDEx datasets.

- GitHub Repository: https://github.com/tsafavi/codex
- Paper: https://arxiv.org/pdf/2009.07810.pdf

Citation:

.. [safavi2020] Safavi, T. & Koutra, D. (2020). `CoDEx: A Comprehensive Knowledge Graph
   Completion Benchmark <http://arxiv.org/abs/2009.07810>`_.  *arXiv* 2009.07810.
"""

from .base import UnpackedRemoteDataSet

BASE_URL = 'https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/'
SMALL_VALID_URL = f'{BASE_URL}/codex-s/valid.txt'
SMALL_TEST_URL = f'{BASE_URL}/codex-s/test.txt'
SMALL_TRAIN_URL = f'{BASE_URL}/codex-s/train.txt'

MEDIUM_VALID_URL = f'{BASE_URL}/codex-m/valid.txt'
MEDIUM_TEST_URL = f'{BASE_URL}/codex-m/test.txt'
MEDIUM_TRAIN_URL = f'{BASE_URL}/codex-m/train.txt'

LARGE_VALID_URL = f'{BASE_URL}/codex-l/valid.txt'
LARGE_TEST_URL = f'{BASE_URL}/codex-l/test.txt'
LARGE_TRAIN_URL = f'{BASE_URL}/codex-l/train.txt'


class CoDeXSmall(UnpackedRemoteDataSet):
    """The CoDeX small dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the CoDeX small dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataSet`.
        """
        # GitHub's raw.githubusercontent.com service rejects requests that are streamable. This is
        # normally the default for all of PyKEEN's remote datasets, so just switch the default here.
        kwargs.setdefault('stream', False)
        super().__init__(
            training_url=SMALL_TRAIN_URL,
            testing_url=SMALL_TEST_URL,
            validation_url=SMALL_VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


class CoDeXMedium(UnpackedRemoteDataSet):
    """The CoDeX medium dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the CoDeX medium dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataSet`.
        """
        kwargs.setdefault('stream', False)  # See comment in CoDeXSmall
        super().__init__(
            training_url=MEDIUM_TRAIN_URL,
            testing_url=MEDIUM_TEST_URL,
            validation_url=MEDIUM_VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


class CoDeXLarge(UnpackedRemoteDataSet):
    """The CoDeX large dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the CoDeX large dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataSet`.
        """
        kwargs.setdefault('stream', False)  # See comment in CoDeXSmall
        super().__init__(
            training_url=LARGE_TRAIN_URL,
            testing_url=LARGE_TEST_URL,
            validation_url=LARGE_VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


def _main():
    for cls in [CoDeXSmall, CoDeXMedium, CoDeXLarge]:
        d = cls()
        d.summarize()


if __name__ == '__main__':
    _main()
