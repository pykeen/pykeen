# -*- coding: utf-8 -*-

"""YAGO3 datasets."""

from typing import Optional

from .base import TarFileRemoteDataSet

__all__ = [
    'YAGO310',
]


class YAGO310(TarFileRemoteDataSet):
    """The YAGO3-10 data set is a subset of YAGO3 that only contains entities with at least 10 relations."""

    def __init__(self, cache_root: Optional[str] = None, **kwargs):
        super().__init__(
            url='https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz',
            relative_training_path='train.txt',
            relative_testing_path='test.txt',
            relative_validation_path='valid.txt',
            cache_root=cache_root,
            **kwargs,
        )
