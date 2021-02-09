# -*- coding: utf-8 -*-

"""The DBpedia datasets from [shi2017b]_.

- GitHub Repository: https://github.com/bxshi/ConMask
- Paper: https://arxiv.org/abs/1711.03438
"""

from .base import UnpackedRemoteDataset

__all__ = [
    'DBpedia50',
]

BASE = 'https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50'
TEST_URL = f'{BASE}/test.txt'
TRAIN_URL = f'{BASE}/train.txt'
VALID_URL = f'{BASE}/valid.txt'


class DBpedia50(UnpackedRemoteDataset):
    """The DBpedia50 dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the DBpedia50 small dataset from [shi2017b]_.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
        """
        # GitHub's raw.githubusercontent.com service rejects requests that are streamable. This is
        # normally the default for all of PyKEEN's remote datasets, so just switch the default here.
        kwargs.setdefault('stream', False)
        super().__init__(
            training_url=TRAIN_URL,
            testing_url=TEST_URL,
            validation_url=VALID_URL,
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )
