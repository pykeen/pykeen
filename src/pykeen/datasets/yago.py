# -*- coding: utf-8 -*-

"""YAGO3 datasets."""

from .base import TarFileRemoteDataset

__all__ = [
    'YAGO310',
]


class YAGO310(TarFileRemoteDataset):
    """The YAGO3-10 dataset is a subset of YAGO3 that only contains entities with at least 10 relations."""

    def __init__(self, create_inverse_triples: bool = False, **kwargs):
        """Initialize the YAGO3-10 dataset.

        :param create_inverse_triples: Should inverse triples be created? Defaults to false.
        :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.TarFileRemoteDataset`.
        """
        super().__init__(
            url='https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz',
            relative_training_path='train.txt',
            relative_testing_path='test.txt',
            relative_validation_path='valid.txt',
            create_inverse_triples=create_inverse_triples,
            **kwargs,
        )
