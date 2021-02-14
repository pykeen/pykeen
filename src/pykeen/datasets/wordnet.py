# -*- coding: utf-8 -*-

"""WordNet datasets."""

import os

from .base import TarFileRemoteDataset

__all__ = [
    'WN18',
    'WN18RR',
]


class WN18(TarFileRemoteDataset):
    """The WN18 dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the WordNet-18 dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.

        .. warning:: This dataset contains testing leakage. Use :class:`WN18RR` instead.
        """
        super().__init__(
            url='https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz',
            relative_training_path=os.path.join('wordnet-mlj12', 'wordnet-mlj12-train.txt'),
            relative_testing_path=os.path.join('wordnet-mlj12', 'wordnet-mlj12-test.txt'),
            relative_validation_path=os.path.join('wordnet-mlj12', 'wordnet-mlj12-valid.txt'),
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )


class WN18RR(TarFileRemoteDataset):
    """The WN18-RR dataset."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the WordNet-18 (RR) dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.
        """
        super().__init__(
            url='https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz',
            relative_training_path='train.txt',
            relative_testing_path='test.txt',
            relative_validation_path='valid.txt',
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )
